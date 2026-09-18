// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: blue; icon-glyph: ticket-alt;

/**
 * JRA 即PAT (IPAT) 自動投票スクリプト (for Scriptable)
 * 予想サイトの「即PAT投票」ボタンから起動します。
 * ログイン -> 通常投票 -> 会場/レース/式別/馬番 -> 金額セット -> 投票確認画面まで自動実行します。
 * 
 * ※ログイン情報はiOSの安全な暗号化領域（Keychain）に初回のみ保存されます。
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
    } catch (e) {}
  }

  if (!params || !params.steps) {
    let a = new Alert();
    a.title = "買い目データが見つかりません";
    a.message = "サイト上の「即PAT」ボタンから実行してください。";
    a.addAction("OK");
    await a.present();
    Script.complete();
    return;
  }

  const { steps, venueName, weekday, unitAmount, totalAmount } = params;
  const uAmount = parseInt(unitAmount) || 100;
  const hundreds = Math.floor(uAmount / 100);
  const tAmount = parseInt(totalAmount) || uAmount;

  // 2. Keychain から IPAT ログイン情報を取得 (初回はプロンプト)
  let userNo = Keychain.contains("ipat_user_no") ? Keychain.get("ipat_user_no") : "";
  let passNo = Keychain.contains("ipat_pass_no") ? Keychain.get("ipat_pass_no") : "";
  let parsNo = Keychain.contains("ipat_pars_no") ? Keychain.get("ipat_pars_no") : "";

  if (!userNo || !passNo || !parsNo) {
    let loginAlert = new Alert();
    loginAlert.title = "即PATログイン情報の設定（初回のみ）";
    loginAlert.message = "JRA即PATの情報を入力してください。\n※端末内の暗号化領域（Keychain）に安全に保存されます。";
    loginAlert.addTextField("加入者番号 (またはINET-ID)", userNo);
    loginAlert.addSecureTextField("暗証番号 (4桁数字)", passNo);
    loginAlert.addSecureTextField("P-ARS番号 (4桁英数字)", parsNo);
    loginAlert.addAction("保存して投票へ進む");
    loginAlert.addCancelAction("キャンセル");
    let resp = await loginAlert.present();
    if (resp === -1) {
      Script.complete();
      return;
    }
    userNo = loginAlert.textFieldValue(0).trim();
    passNo = loginAlert.textFieldValue(1).trim();
    parsNo = loginAlert.textFieldValue(2).trim();

    if (!userNo || !passNo || !parsNo) {
      let errA = new Alert();
      errA.title = "入力エラー";
      errA.message = "未入力の項目があります。";
      await errA.present();
      Script.complete();
      return;
    }

    Keychain.set("ipat_user_no", userNo);
    Keychain.set("ipat_pass_no", passNo);
    Keychain.set("ipat_pars_no", parsNo);
  }

  // 3. JRA IPAT スマホ版を開く
  let wv = new WebView();
  let presentPromise = wv.present(true);
  await wv.loadURL("https://www.ipat.jra.go.jp/sp/");

  // 4. ログイン実行
  let loginScript = `
  (function() {
    var u = ${JSON.stringify(userNo)};
    var p = ${JSON.stringify(passNo)};
    var r = ${JSON.stringify(parsNo)};

    var elU = document.getElementById("userid") || document.querySelector("input[name='i']");
    var elP = document.getElementById("password") || document.querySelector("input[name='p']");
    var elR = document.getElementById("pars") || document.querySelector("input[name='r']");

    if (elU && elP && elR) {
      elU.value = u;
      elP.value = p;
      elR.value = r;
      if (typeof ToSPMenu === "function") {
        ToSPMenu();
      } else {
        var form = document.querySelector("form");
        if (form) form.submit();
      }
    }
  })();
  `;
  await wv.evaluateJavaScript(loginScript, false);
  await wv.waitForLoad();

  // 5. ログイン後メニューから「通常投票」へ遷移
  let toVoteScript = `
  (function() {
    var a = document.querySelector("a.ico_regular");
    if (a) {
      a.click();
    } else if (typeof ToSPBet === "function") {
      ToSPBet(0);
    }
  })();
  `;
  await wv.evaluateJavaScript(toVoteScript, false);
  await wv.waitForLoad();

  // ページ初期化を少し待機
  await new Promise(res => Timer.schedule(600, false, res));

  // 6. 通常投票画面での自動入力 & 金額セット & 投票確認画面へ遷移
  let runnerScript = `
  (function() {
    var s = ${JSON.stringify(steps)};
    var vn = ${JSON.stringify(venueName || "")};
    var wd = ${JSON.stringify(weekday || "")};
    var hundreds = ${hundreds};
    var totalAmount = ${tAmount};
    var passNo = ${JSON.stringify(passNo)};
    var sn = {"1":"単勝","2":"複勝","3":"枠連","4":"馬連","5":"ワイド","6":"馬単","7":"3連複","8":"3連単"};
    var i = 0, r = 0, d = false, T = Date.now();

    function dg(m, color) {
      var x = document.getElementById("ipat-diag");
      if (!x) {
        x = document.createElement("div");
        x.id = "ipat-diag";
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:100000;background:" + (color || "rgba(79,70,229,0.95)") + ";color:#ffffff;font-size:12px;font-weight:bold;padding:8px 12px;pointer-events:none;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.3);";
        document.body.appendChild(x);
      }
      x.innerText = m;
      if (color) x.style.background = color;
    }

    function tp(e) {
      var rect = e.getBoundingClientRect();
      var x = rect.left + rect.width / 2;
      var y = rect.top + rect.height / 2;
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

    function setAmountAndSend() {
      dg("💰 金額を入力中 (1点: " + (hundreds * 100) + "円)...");
      var kinInput = document.querySelector("#kin .amount input, input.amount");
      if (kinInput) {
        kinInput.value = String(hundreds);
        if (kinInput.onchange) kinInput.onchange();
      } else if (typeof VMA === "function") {
        VMA(String(hundreds));
      }

      setTimeout(function() {
        dg("セット処理を実行中...");
        if (typeof SetBet === "function") {
          SetBet(0);
        } else {
          var k = ["セット", "次へ", "決定"];
          var a = document.querySelectorAll("a,button");
          for (var j = 0; j < a.length; j++) {
            for (var l = 0; l < k.length; l++) {
              if (a[j].textContent.indexOf(k[l]) >= 0) { tp(a[j]); break; }
            }
          }
        }

        setTimeout(function() {
          dg("確認画面へ移動中...");
          if (typeof ToSend === "function") {
            ToSend();
          } else {
            var sendBtn = Array.from(document.querySelectorAll("a,button")).find(function(b) {
              return b.textContent.includes("入力終了") || b.textContent.includes("投票確認");
            });
            if (sendBtn) tp(sendBtn);
          }

          // 確認画面での入力処理
          pollConfirmScreen();
        }, 1200);
      }, 800);
    }

    function pollConfirmScreen() {
      var pollCount = 0;
      var timer = setInterval(function() {
        pollCount++;
        var inputs = document.querySelectorAll("input");
        var hasPassword = document.querySelector("input[type='password'], #password");
        
        if (hasPassword || pollCount > 30) {
          clearInterval(timer);
          fillConfirmForm();
        }
      }, 400);
    }

    function fillConfirmForm() {
      // 合計金額入力
      var inputs = document.querySelectorAll("input");
      for (var j = 0; j < inputs.length; j++) {
        var inp = inputs[j];
        var itype = (inp.type || "").toLowerCase();
        var iid = (inp.id || "").toLowerCase();
        var iname = (inp.name || "").toLowerCase();
        if ((itype === "tel" || itype === "text" || itype === "number") && iid.indexOf("pass") < 0 && iname.indexOf("pass") < 0) {
          inp.value = String(totalAmount);
          if (inp.onchange) inp.onchange();
          break;
        }
      }

      // 暗証番号入力
      var passInput = document.querySelector("input[type='password'], #password");
      if (passInput) {
        passInput.value = passNo;
        if (passInput.onchange) passInput.onchange();
      }

      dg("✅ セット完了！内容を確認し、よろしければ【投票】を押してください", "rgba(16,185,129,0.95)");
    }

    function nx() {
      try {
        if (Date.now() - T > 25000) { dg("⚠️ タイムアウトしました"); return; }
        var p = "";
        if (document.getElementById("jyo")) p = "V";
        else if (document.getElementById("race")) p = "R";
        else if (document.getElementById("siki")) p = "S";
        else if (document.getElementById("hou")) p = "M";

        if (i >= s.length) {
          setAmountAndSend();
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

        dg("🏇 即PAT自動選択中... Step " + (i + 1) + "/" + s.length);

        var okP = (i === 0 && (p === "V" || p === "" || r > 1)) ||
                  (i === 1 && (p === "R" || p === "V" || p === "" || r > 1)) ||
                  (i === 2 && (p === "S" || r > 1)) ||
                  (i === 3 && (p === "M" || p === "S" || r > 1)) ||
                  (i > 3);

        if (okP) {
          if (i === 0) {
            var bs = document.querySelectorAll("a,button");
            for (var k2 = 0; k2 < bs.length; k2++) {
              if (bs[k2].getBoundingClientRect().width <= 4) continue;
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
                  if (es[j].getBoundingClientRect().width > 3) {
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
                if (es[j].getBoundingClientRect().width > 3) {
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
                if (bs[k2].getBoundingClientRect().width <= 4) continue;
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
        dg("エラー: " + e.message);
      }
    }
    nx();
  })();
  `;

  await wv.evaluateJavaScript(runnerScript, false);

  // 7. WebViewの終了待機
  await presentPromise;
  Script.complete();
}

await main();
