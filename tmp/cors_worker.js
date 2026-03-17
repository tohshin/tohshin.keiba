export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // 1. クエリパラメータ（?pid=...等）を含めてターゲットURLを再構築する
    // url.pathname.slice(1) は "https://race.sp.netkeiba.com/"
    // url.search は "?pid=race_result&race_id=..."
    let targetUrl = url.searchParams.get("url") || (url.pathname.slice(1) + url.search);

    if (!targetUrl || targetUrl === "/") {
      return new Response("CORS Proxy: Please provide a target URL.", { status: 400 });
    }

    if (!targetUrl.startsWith("http")) {
      targetUrl = "https://" + targetUrl;
    }

    try {
      // 2. Netkeiba のチェックを回避するため User-Agent を設定する
      const newRequestHeaders = new Headers(request.headers);
      newRequestHeaders.set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
      
      // 不要または干渉する可能性のあるヘッダーを削除
      newRequestHeaders.delete("Host");

      console.log("Proxying to:", targetUrl);

      const response = await fetch(targetUrl, {
        method: request.method,
        headers: newRequestHeaders,
        redirect: "follow"
      });

      // 3. レスポンスに CORS ヘッダーを付与
      const newHeaders = new Headers(response.headers);
      newHeaders.set("Access-Control-Allow-Origin", "*");
      newHeaders.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      newHeaders.set("Access-Control-Allow-Headers", "*");
      
      // Netkeiba の Location ヘッダー等のリダイレクト先がプロキシを介さなくなるのを防ぐため、
      // 必要に応じてリダイレクト処理を fetch 側に任せる（redirect: "follow"済み）

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders
      });
    } catch (e) {
      return new Response("Proxy Error: " + e.message, { status: 500 });
    }
  }
};
