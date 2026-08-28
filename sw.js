const CACHE_NAME = 'keiba-ai-v2';
const CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 1日 (24時間)

// インストール時にコアアセットをプリキャッシュ
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll([
                './',
                './index.html'
            ]);
        }).then(() => self.skipWaiting())
    );
});

// アクティベート時に古いキャッシュを破棄
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((name) => {
                    if (name !== CACHE_NAME) {
                        return caches.delete(name);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// キャッシュに保存（タイムスタンプヘッダーを付与）
async function saveToCache(request, response) {
    if (!response || response.status !== 200 || response.type === 'opaque') {
        return response;
    }

    try {
        const cache = await caches.open(CACHE_NAME);
        const headers = new Headers(response.headers);
        headers.set('x-sw-cached-time', Date.now().toString());

        const responseToCache = new Response(await response.clone().blob(), {
            status: response.status,
            statusText: response.statusText,
            headers: headers
        });

        // URLクエリ（?t=...）がある場合も正規化して保存
        await cache.put(request, responseToCache);
    } catch (e) {
        console.warn('[SW] Cache save error:', e);
    }
    return response;
}

// ネットワーク優先 (Network First) + 24時間キャッシュフォールバック
self.addEventListener('fetch', (event) => {
    // GETリクエスト以外は無視
    if (event.request.method !== 'GET') return;

    // HTTP / HTTPS リクエストのみ対象
    const url = new URL(event.request.url);
    if (!url.protocol.startsWith('http')) return;

    event.respondWith(
        (async () => {
            try {
                // 1. まずネットワークから最新データを取得を試みる
                const networkResponse = await fetch(event.request);
                if (networkResponse && networkResponse.status === 200) {
                    // バックグラウンドでキャッシュ保存
                    saveToCache(event.request, networkResponse.clone());
                    return networkResponse;
                }
            } catch (networkError) {
                console.log('[SW] Network fetch failed, falling back to cache:', event.request.url);
            }

            // 2. ネットワーク失敗時（オフライン時）、キャッシュを検索
            // ignoreSearch: true で ?t=... 等のタイムスタンプパラメータ付与時もキャッシュから取得
            let cachedResponse = await caches.match(event.request, { ignoreSearch: true });
            
            if (!cachedResponse && url.pathname.endsWith('/')) {
                cachedResponse = await caches.match('./index.html');
            }

            if (cachedResponse) {
                const cachedTimeStr = cachedResponse.headers.get('x-sw-cached-time');
                if (cachedTimeStr) {
                    const cachedTime = parseInt(cachedTimeStr, 10);
                    const age = Date.now() - cachedTime;

                    if (age > CACHE_TTL_MS) {
                        // 1日以上経過したキャッシュの場合
                        console.warn('[SW] Cache expired (>1 day):', event.request.url);
                        // オフラインでもデータが無いよりは閲覧できた方が良いためデータを返却するが、ヘッダーに期限切れフラグを付与
                        const headers = new Headers(cachedResponse.headers);
                        headers.set('x-sw-cache-expired', 'true');
                        return new Response(await cachedResponse.blob(), {
                            status: cachedResponse.status,
                            statusText: cachedResponse.statusText,
                            headers: headers
                        });
                    }
                }
                return cachedResponse;
            }

            // キャッシュにも無い場合
            return new Response('Offline and no cache available', {
                status: 503,
                statusText: 'Service Unavailable',
                headers: new Headers({ 'Content-Type': 'text/plain; charset=utf-8' })
            });
        })()
    );
});
