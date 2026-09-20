// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: green; icon-glyph: horse;

/**
 * JRAスマッピー自動入力スクリプト (for Scriptable)
 * 複数買い目の連続入力・状態機械対応版
 * 
 * 【主な改修点】
 * 1. 入力データ正規化: 新形式 {bets:[{venueName,weekday,steps}]} と 旧形式 {steps,venueName,weekday} を受付け
 * 2. 状態機械 (State Machine): BOOT / START / GOTO_JYO / RUN_BET / WAIT_USER / NEXT / DONE
 * 3. 買い目ごとのRUN_BET実行: 買い目ごとに25秒タイムアウト、steps完了後は WAIT_USER へ
 * 4. WAIT_USER待機: タイムアウトなし。ユーザーが馬番・金額を入力して投票一覧(#toui)へ進み、
 *    ul.voteList li の件数が増加したら次へ
 * 5. GOTO_JYO遷移: #fromjyo_top a（非表示なら #fromjyo_under a）をクリックして #jyo へ遷移
 * 6. ページ判定: id (#jyo,#race,#siki,#hou,#toui) を優先し、フォールバックは else if でテキスト判定
 * 7. 診断表示: 画面上部バーに「k/N件目 状態」と詳細をリアルタイム表示
 * 8. Scriptable制約準拠: WebView.evaluateJavaScript を1回呼び出し、内部ループで完結
 * 
 * 【動作確認手順】
 * (1) 単一買い目（旧形式互換）テスト:
 *     クリップボードに {"steps":["阪神","8","8","0"],"venueName":"阪神","weekday":"日"} をコピーしてScriptableで実行。
 *     場名・レース・式別・方式が選択され、馬番選択画面で手動入力待ち（WAIT_USER）になることを確認。
 * (2) 複数買い目テスト:
 *     クリップボードに {"bets":[{"venueName":"阪神","weekday":"日","steps":["阪神","8","8","0"]},{"venueName":"阪神","weekday":"日","steps":["阪神","3","1","0"]}]}
 *     をコピーしてScriptableで実行。
 *     1件目の選択完了後、馬番・金額を入力して「投票一覧へ」進むと、自動的に「場名から続けて入力」が押されて
 *     2件目のステップが自動実行されることを確認。
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

  // [改修点1] 入力JSONの正規化
  // {bets:[{venueName,weekday,steps}]} 形式と 旧形式 {steps,venueName,weekday} の両方に対応
  let bets = [];
  if (params) {
    if (params.bets && Array.isArray(params.bets)) {
      bets = params.bets;
    } else if (params.steps) {
      bets = [{
        steps: params.steps,
        venueName: params.venueName || "",
        weekday: params.weekday || ""
      }];
    }
  }

  if (!bets || bets.length === 0) {
    let a = new Alert();
    a.title = "データが見つかりません";
    a.message = "買い目データが渡されていません。競馬予想サイトの「Scriptableで投票」ボタンを押してください。";
    a.addAction("OK");
    await a.present();
    Script.complete();
    return;
  }

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

  // 4. pw_982_i.cgi 上での状態機械オートメーション
  // [改修点8] WebView.evaluateJavaScript を1回呼び出し、内部ループで完結 (ES5準拠)
  let runnerScript = `
  (function() {
    var bets = ${JSON.stringify(bets)};
    var sn = {"1":"単勝","2":"複勝","3":"枠連","4":"馬連","5":"ワイド","6":"馬単","7":"3連複","8":"3連単"};

    // [改修点2] 状態機械: BOOT / START / GOTO_JYO / RUN_BET / WAIT_USER / NEXT / DONE
    var state = "BOOT";
    var betIndex = 0;
    var stepIndex = 0;
    var retryCount = 0;
    var startVoteCount = 0;
    var betStartTime = 0;

    // [改修点7] 画面上部 診断バー表示
    function dg(m) {
      var x = document.getElementById("smappy-diag");
      if (!x) {
        x = document.createElement("div");
        x.id = "smappy-diag";
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:100000;background:rgba(0,0,0,0.9);color:#0f0;font-size:11px;padding:4px 8px;pointer-events:none;font-family:sans-serif;line-height:1.3;box-sizing:border-box;word-break:break-all;";
        document.body.appendChild(x);
      }
      x.innerText = m;
    }

    // 診断表示用フォーマット: 「k/N件目 状態」を付与
    function fmtDiag(detail) {
      var k = betIndex + 1;
      var n = bets.length;
      return "[" + k + "/" + n + "件目 " + state + "] " + (detail || "");
    }

    // [流用] タッチ/マウス/クリック発火ロジック
    function tp(e) {
      if (!e) return;
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

    // [流用] 馬番選択へ進む確定ボタン等のクリック
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

    // [改修点6] ページ判定: id (#jyo,#race,#siki,#hou,#toui) を優先し、フォールバックは else if
    function getPage() {
      // 1. アクティブページ (.ui-page-active) の id を最優先
      var act = document.querySelector(".ui-page-active");
      if (act && act.id) {
        if (act.id === "jyo") return "jyo";
        if (act.id === "race") return "race";
        if (act.id === "siki") return "siki";
        if (act.id === "hou") return "hou";
        if (act.id === "toui") return "toui";
      }
      // 2. DOM全体の id で判定
      if (document.getElementById("jyo")) return "jyo";
      else if (document.getElementById("race")) return "race";
      else if (document.getElementById("siki")) return "siki";
      else if (document.getElementById("hou")) return "hou";
      else if (document.getElementById("toui")) return "toui";
      // 3. テキストによるフォールバック (else if)
      else {
        var c = (document.body ? (document.body.innerText || "") : "");
        if (c.indexOf("投票一覧") >= 0 || c.indexOf("投票内容") >= 0) return "toui";
        else if (c.indexOf("競馬場") >= 0 || c.indexOf("会場") >= 0 || c.indexOf("開催") >= 0) return "jyo";
        else if (c.indexOf("レース") >= 0 || c.indexOf("回次") >= 0) return "race";
        else if (c.indexOf("式別") >= 0) return "siki";
        else if (c.indexOf("方式") >= 0) return "hou";
      }
      return "";
    }

    // [改修点4] 投票一覧 (ul.voteList li) の件数を取得
    function getVoteCount() {
      var list = document.querySelectorAll("ul.voteList li");
      return list ? list.length : 0;
    }

    // [改修点5] #fromjyo_top a（非表示なら #fromjyo_under a）をクリック
    function clickFromJyo() {
      var topBtn = document.querySelector("#fromjyo_top a");
      if (topBtn) {
        var r = topBtn.getBoundingClientRect();
        var st = window.getComputedStyle ? window.getComputedStyle(topBtn) : null;
        var isVis = r.width > 0 && r.height > 0 && (!st || (st.display !== "none" && st.visibility !== "hidden"));
        if (isVis) {
          tp(topBtn);
          return true;
        }
      }
      var underBtn = document.querySelector("#fromjyo_under a");
      if (underBtn) {
        var ur = underBtn.getBoundingClientRect();
        if (ur.width > 0 && ur.height > 0) {
          tp(underBtn);
          return true;
        }
      }
      return false;
    }

    // 状態機械メインループ
    function loop() {
      try {
        var curPage = getPage();

        switch (state) {
          case "BOOT":
            dg(fmtDiag("初期化中..."));
            // ページまたは選択要素が存在すれば START へ
            if (curPage || document.querySelector("a,button")) {
              state = "START";
              setTimeout(loop, 200);
            } else {
              setTimeout(loop, 300);
            }
            break;

          case "START":
            betIndex = 0;
            stepIndex = 0;
            retryCount = 0;
            startVoteCount = getVoteCount();
            betStartTime = Date.now();
            state = "RUN_BET";
            dg(fmtDiag("買い目入力開始"));
            setTimeout(loop, 100);
            break;

          case "GOTO_JYO":
            dg(fmtDiag("競馬場画面へ遷移中 (page:" + (curPage || "none") + ")"));
            if (curPage === "jyo") {
              // #jyo に到達したら次の買い目の RUN_BET を開始
              stepIndex = 0;
              retryCount = 0;
              betStartTime = Date.now();
              state = "RUN_BET";
              setTimeout(loop, 400);
            } else {
              // #fromjyo_top a（非表示なら #fromjyo_under a）をクリック
              var clicked = clickFromJyo();
              setTimeout(loop, clicked ? 600 : 300);
            }
            break;

          case "RUN_BET":
            // [改修点4] RUN_BET は買い目ごとに25秒タイムアウト
            if (Date.now() - betStartTime > 25000) {
              dg(fmtDiag("タイムアウト(25秒) 手動で操作してください"));
              state = "WAIT_USER";
              setTimeout(loop, 1000);
              return;
            }

            var curBet = bets[betIndex];
            var s = curBet.steps || [];
            var vn = curBet.venueName || "";

            // steps を全て完了したら、馬番・金額の手動入力待ち(WAIT_USER)へ
            if (stepIndex >= s.length) {
              dg(fmtDiag("steps完了 -> 馬番・金額を入力し投票一覧へ"));
              cf(); // 確定/セット/次へ ボタンをクリック
              state = "WAIT_USER";
              setTimeout(loop, 600);
              return;
            }

            var v = s[stepIndex];
            var f = false;
            var vs = [v];
            var n = parseInt(v, 10);
            if (!isNaN(n)) {
              if (stepIndex === 1) {
                vs = [String(n - 1), (n - 1 < 10 ? "0" + (n - 1) : String(n - 1))];
              } else {
                vs = [v, String(n), (n < 10 ? "0" + n : String(n)), String(n - 1), (n - 1 < 10 ? "0" + (n - 1) : String(n - 1))];
              }
            }

            dg(fmtDiag("S" + stepIndex + ":" + v + " r:" + retryCount + " p:" + curPage));

            var okP = (stepIndex === 0 && (curPage === "jyo" || curPage === "" || retryCount > 1)) ||
                      (stepIndex === 1 && (curPage === "race" || curPage === "jyo" || curPage === "" || retryCount > 1)) ||
                      (stepIndex === 2 && (curPage === "siki" || retryCount > 1)) ||
                      (stepIndex === 3 && (curPage === "hou" || curPage === "siki" || retryCount > 1)) ||
                      (stepIndex > 3);

            if (okP) {
              if (stepIndex === 0) {
                // 競馬場名の選択
                var bs = document.querySelectorAll("a,button");
                for (var k2 = 0; k2 < bs.length; k2++) {
                  var b2 = bs[k2].getBoundingClientRect();
                  if (b2.width <= 4 || b2.height <= 4 || bs[k2].classList.contains("disabled")) continue;
                  var t = (bs[k2].innerText || bs[k2].textContent || "").trim();
                  if (vn && t.indexOf(vn) >= 0) {
                    tp(bs[k2]);
                    stepIndex++;
                    retryCount = 0;
                    f = true;
                    setTimeout(loop, 450);
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
                        stepIndex++;
                        retryCount = 0;
                        f = true;
                        setTimeout(loop, 450);
                        break;
                      }
                    }
                    if (f) break;
                  }
                }
              } else {
                // レース・式別・方式等の選択
                for (var k = 0; k < vs.length; k++) {
                  var es = document.querySelectorAll("a[data-value='" + vs[k] + "'],button[data-value='" + vs[k] + "']");
                  for (var j = 0; j < es.length; j++) {
                    var b = es[j].getBoundingClientRect();
                    if (b.width > 3 && b.height > 3) {
                      tp(es[j]);
                      stepIndex++;
                      retryCount = 0;
                      f = true;
                      setTimeout(loop, 450);
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
                    if (stepIndex === 1 && (t === v + "R" || t === v + "レース" || t.indexOf(v + "R") >= 0)) {
                      tp(bs[k2]);
                      stepIndex++;
                      retryCount = 0;
                      f = true;
                      setTimeout(loop, 450);
                      break;
                    }
                    if (stepIndex === 2 && sn[v] && t.indexOf(sn[v]) >= 0) {
                      tp(bs[k2]);
                      stepIndex++;
                      retryCount = 0;
                      f = true;
                      setTimeout(loop, 450);
                      break;
                    }
                  }
                }
              }
            }

            if (!f) {
              retryCount++;
              setTimeout(loop, 200);
            }
            break;

          case "WAIT_USER":
            // [改修点4] タイムアウトなし。ユーザーが手動で馬番・金額を入力し投票一覧(#toui)へ進むのを待機
            var currentVotes = getVoteCount();
            dg(fmtDiag("馬番・金額入力待ち (件数: " + currentVotes + " / 開始時: " + startVoteCount + ")"));

            // 投票一覧(#toui)へ到達し、投票件数が開始時より増加したか判定
            if (curPage === "toui" && currentVotes > startVoteCount) {
              if (betIndex + 1 < bets.length) {
                // まだ次の買い目がある場合 -> NEXT へ
                state = "NEXT";
                setTimeout(loop, 300);
              } else {
                // 全買い目完了 -> DONE へ
                state = "DONE";
                setTimeout(loop, 100);
              }
            } else {
              setTimeout(loop, 500);
            }
            break;

          case "NEXT":
            betIndex++;
            startVoteCount = getVoteCount(); // 次の買い目の判定基準件数を更新
            state = "GOTO_JYO";
            dg(fmtDiag("次の買い目へ遷移準備"));
            setTimeout(loop, 300);
            break;

          case "DONE":
            dg(fmtDiag("全" + bets.length + "件の入力完了！「投票用QR表示」を押してください"));
            break;
        }
      } catch (e) {
        dg("ERR: " + e.message);
        setTimeout(loop, 1000);
      }
    }

    loop();
  })();
  `;

  await wv.evaluateJavaScript(runnerScript, false);

  // 5. WebViewが閉じられるのを待つ
  await presentPromise;
  Script.complete();
}

await main();
