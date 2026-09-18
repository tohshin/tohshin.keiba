// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: green; icon-glyph: horse;

/**
 * JRAスマッピー自動入力スクリプト (for Scriptable)
 * https://qrcode.jra.go.jp/ から開始して通常投票(pw_982_i.cgi)へ自動遷移し、
 * 買い目を自動選択してQRコード作成・金額画面を表示します。
 */

async function main() {
  // 1. パラメータの取得 (URLクエリ または クリップボード)
  let rawData = null;
  if (args.queryParameters && args.queryParameters.data) {
    rawData = args.queryParameters.data;
  }
  if (!rawData) {
    rawData = Pasteboard.paste();
  }

  let params = null;
  if (rawData) {
    try {
      params = JSON.parse(rawData);
    } catch (e) {
      // JSONパース失敗
    }
  }

  if (!params || !params.steps) {
    let a = new Alert();
    a.title = "データが見つかりません";
    a.message = "買い目データが渡されていません。競馬予想サイトの「Scriptableで投票」ボタンを押してください。";
    a.addAction("OK");
    await a.present();
    Script.complete();
    return;
  }

  const { steps, venueName, weekday } = params;

  // 2. JRAスマッピーのトップページを開く
  let topUrl = "https://qrcode.jra.go.jp/";
  let wv = new WebView();
  let presentPromise = wv.present(true);
  await wv.loadURL(topUrl);

  // 3. トップページから「通常投票」への自動遷移
  let currentUrl = await wv.evaluateJavaScript("window.location.href", false);
  if (!currentUrl || !currentUrl.includes("pw_982_i.cgi")) {
    // ToQRBet または 通常投票リンクが準備できるまで待機（最大3秒）
    for (let k = 0; k < 15; k++) {
      let ready = await wv.evaluateJavaScript("typeof ToQRBet === 'function' || !!document.querySelector('a.ico_regular') || !!document.FORM0", false);
      if (ready) break;
      await new Promise(r => Timer.schedule(200, false, r));
    }

    // 通常投票へ遷移実行
    await wv.evaluateJavaScript(`
      (function() {
        if (typeof ToQRBet === "function") {
          ToQRBet();
        } else {
          var a = document.querySelector("a.ico_regular");
          if (a) {
            a.click();
          } else if (document.FORM0) {
            document.FORM0.action = "pw_982_i.cgi#";
            document.FORM0.submit();
          }
        }
      })();
    `, false);

    // pw_982_i.cgi へのページ遷移完了を待機
    await wv.waitForLoad();
  }

  // ページ初期化を少し待機
  await new Promise(r => Timer.schedule(500, false, r));

  // 4. pw_982_i.cgi 上での自動操作（前と同じ動き）
  let runnerScript = `
  (function() {
    var s = ${JSON.stringify(steps)};
    var vn = ${JSON.stringify(venueName || "")};
    var wd = ${JSON.stringify(weekday || "")};
    var sn = {"1":"単勝","2":"複勝","3":"枠連","4":"馬連","5":"ワイド","6":"馬単","7":"3連複","8":"3連単"};
    var i = 0, r = 0, d = false, T = Date.now();

    function dg(m) {
      var x = document.getElementById("smappy-diag");
      if (!x) {
        x = document.createElement("div");
        x.id = "smappy-diag";
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:100000;background:rgba(0,0,0,0.9);color:#0f0;font-size:10px;padding:4px;pointer-events:none;";
        document.body.appendChild(x);
      }
      x.innerText = m;
    }

    function fi(ok) {
      if (d) return;
      d = true;
      dg("FINISH:" + ok);
    }

    function tp(e) {
      var r = e.getBoundingClientRect();
      var x = r.left + r.width / 2;
      var y = r.top + r.height / 2;
      var o = {bubbles:true, cancelable:true, clientX:x, clientY:y, view:window};
      try {
        var t = new Touch({identifier:Date.now(), target:e, clientX:x, clientY:y, radiusX:2, radiusY:2});
        var to = {bubbles:true, cancelable:true, touches:[t], targetTouches:[t], changedTouches:[t], view:window};
        e.dispatchEvent(new TouchEvent("touchstart", to));
        e.dispatchEvent(new TouchEvent("touchend", to));
      } catch(err) {}
      e.dispatchEvent(new MouseEvent("mousedown", o));
      e.dispatchEvent(new MouseEvent("mouseup", o));
      e.dispatchEvent(new MouseEvent("click", o));
      try { e.click(); } catch(err) {}
    }

    function cf() {
      var k = ["金額", "セット", "次へ", "決定"];
      var a = document.querySelectorAll("a,button");
      for (var j = 0; j < a.length; j++) {
        var b = a[j].getBoundingClientRect();
        if (b.width > 0 && b.height > 0) {
          for (var l = 0; l < k.length; l++) {
            if (a[j].textContent.indexOf(k[l]) >= 0) {
              tp(a[j]);
              return;
            }
          }
        }
      }
    }

    function nx() {
      try {
        if (Date.now() - T > 25000) { fi(false); return; }
        var p = "";
        if (document.getElementById("jyo")) p = "V";
        else if (document.getElementById("race")) p = "R";
        else if (document.getElementById("siki")) p = "S";
        else if (document.getElementById("hou")) p = "M";
        else {
          var c = (document.body.innerText || "");
          if (c.indexOf("会場") >= 0 || c.indexOf("開催") >= 0) p = "V";
          if (c.indexOf("レース") >= 0 || c.indexOf("回次") >= 0) p = "R";
          if (c.indexOf("式別") >= 0) p = "S";
          if (c.indexOf("方式") >= 0) p = "M";
        }
        if (i >= s.length) {
          dg("Done");
          cf();
          fi(true);
          return;
        }
        var v = s[i];
        var f = false;
        var vs = [v];
        var n = parseInt(v);
        if (!isNaN(n)) {
          if (i === 1) {
            vs = [String(n - 1), (n - 1 < 10 ? "0" + (n - 1) : String(n - 1))];
          } else {
            vs = [v, String(n), (n < 10 ? "0" + n : String(n)), String(n - 1), (n - 1 < 10 ? "0" + (n - 1) : String(n - 1))];
          }
        }
        dg("S" + i + ":" + v + " r:" + r + " p:" + p);
        var okP = (i === 0 && (p === "V" || p === "" || r > 1)) ||
                  (i === 1 && (p === "R" || p === "V" || p === "" || r > 1)) ||
                  (i === 2 && (p === "S" || r > 1)) ||
                  (i === 3 && (p === "M" || p === "S" || r > 1)) ||
                  (i > 3);
        if (okP) {
          if (i === 0) {
            var bs = document.querySelectorAll("a,button");
            for (var k2 = 0; k2 < bs.length; k2++) {
              var b2 = bs[k2].getBoundingClientRect();
              if (b2.width <= 4 || b2.height <= 4 || bs[k2].classList.contains("disabled")) continue;
              var t = (bs[k2].innerText || bs[k2].textContent || "").trim();
              if (vn && t.indexOf(vn) >= 0) {
                tp(bs[k2]);
                i++;
                r = 0;
                setTimeout(nx, 450);
                f = true;
                break;
              }
            }
            if (!f) {
              for (var k = 0; k < vs.length; k++) {
                var es = document.querySelectorAll("a[data-value='" + vs[k] + "'],button[data-value='" + vs[k] + "']");
                for (var j = 0; j < es.length; j++) {
                  var b = es[j].getBoundingClientRect();
                  if (b.width > 3 && b.height > 3) {
                    tp(es[j]);
                    i++;
                    r = 0;
                    setTimeout(nx, 450);
                    f = true;
                    break;
                  }
                }
                if (f) break;
              }
            }
          } else {
            for (var k = 0; k < vs.length; k++) {
              var es = document.querySelectorAll("a[data-value='" + vs[k] + "'],button[data-value='" + vs[k] + "']");
              for (var j = 0; j < es.length; j++) {
                var b = es[j].getBoundingClientRect();
                if (b.width > 3 && b.height > 3) {
                  tp(es[j]);
                  i++;
                  r = 0;
                  setTimeout(nx, 450);
                  f = true;
                  break;
                }
              }
              if (f) break;
            }
            if (!f) {
              var bs = document.querySelectorAll("a,button");
              for (var k2 = 0; k2 < bs.length; k2++) {
                var b2 = bs[k2].getBoundingClientRect();
                if (b2.width <= 4 || b2.height <= 4) continue;
                var t = (bs[k2].innerText || bs[k2].textContent || "").trim();
                if (i === 1 && (t === v + "R" || t === v + "レース" || t.indexOf(v + "R") >= 0)) {
                  tp(bs[k2]);
                  i++;
                  r = 0;
                  setTimeout(nx, 450);
                  f = true;
                  break;
                }
                if (i === 2 && sn[v] && t.indexOf(sn[v]) >= 0) {
                  tp(bs[k2]);
                  i++;
                  r = 0;
                  setTimeout(nx, 450);
                  f = true;
                  break;
                }
              }
            }
          }
        }
        if (!f) {
          r++;
          setTimeout(nx, 200);
        }
      } catch (e) {
        dg("E:" + e.message);
        fi(false);
      }
    }
    nx();
  })();
  `;

  await wv.evaluateJavaScript(runnerScript, false);

  // 5. WebViewが閉じられるのを待つ
  await presentPromise;
  Script.complete();
}

await main();
