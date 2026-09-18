// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: purple; icon-glyph: id-card;

/**
 * JRA UMACA スマート自動投票スクリプト (for Scriptable)
 * 予想サイトの「UMACA」ボタンから起動します。
 * ログイン -> 通常投票 -> 会場/レース/式別/馬番 -> 金額セット -> 投票確認画面まで自動実行します。
 * 
 * ※ログイン情報（カード番号・生年月日・暗証番号）はiOSの安全な暗号化領域（Keychain）に初回のみ保存されます。
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
    a.message = "サイト上の「UMACA」ボタンから実行してください。";
    a.addAction("OK");
    await a.present();
    Script.complete();
    return;
  }

  const { steps, venueName, weekday, unitAmount, totalAmount } = params;
  const uAmount = parseInt(unitAmount) || 100;
  const hundreds = Math.floor(uAmount / 100);
  const tAmount = parseInt(totalAmount) || uAmount;

  // 2. Keychain から UMACA ログイン情報を取得 (初回はプロンプト)
  let cardNo = Keychain.contains("umaca_card_no") ? Keychain.get("umaca_card_no") : "";
  let birthDay = Keychain.contains("umaca_birth_day") ? Keychain.get("umaca_birth_day") : "";
  let passNo = Keychain.contains("umaca_pass_no") ? Keychain.get("umaca_pass_no") : "";

  if (!cardNo || !birthDay || !passNo) {
    let loginAlert = new Alert();
    loginAlert.title = "UMACAログイン情報の設定（初回のみ）";
    loginAlert.message = "JRA UMACAカードの情報を入力してください。\n※端末内の暗号化領域（Keychain）に安全に保存されます。";
    loginAlert.addTextField("UMACAカード番号 (12桁数字)", cardNo);
    loginAlert.addTextField("生年月日 (8桁: 例 19900101)", birthDay);
    loginAlert.addSecureTextField("暗証番号 (4桁数字)", passNo);
    loginAlert.addAction("保存して投票へ進む");
    loginAlert.addCancelAction("キャンセル");
    let resp = await loginAlert.present();
    if (resp === -1) {
      Script.complete();
      return;
    }
    cardNo = loginAlert.textFieldValue(0).trim();
    birthDay = loginAlert.textFieldValue(1).trim();
    passNo = loginAlert.textFieldValue(2).trim();

    if (!cardNo || !birthDay || !passNo) {
      let errA = new Alert();
      errA.title = "入力エラー";
      errA.message = "未入力の項目があります。";
      await errA.present();
      Script.complete();
      return;
    }

    Keychain.set("umaca_card_no", cardNo);
    Keychain.set("umaca_birth_day", birthDay);
    Keychain.set("umaca_pass_no", passNo);
  }

  // 3. JRA UMACAスマート投票サイトを開く
  let wv = new WebView();
  let presentPromise = wv.present(true);
  await wv.loadURL("https://www.ipat.jra.go.jp/sp/umaca/");

  // 4. ログイン実行
  let loginScript = `
  (function() {
    var c = ${JSON.stringify(cardNo)};
    var b = ${JSON.stringify(birthDay)};
    var p = ${JSON.stringify(passNo)};

    // フォーム要素の探索（nameやID、順序で柔軟にフォールバック）
    var inps = Array.from(document.querySelectorAll("input"));
    var elC = document.getElementById("cardno") || document.querySelector("input[name='c']") || inps.find(i => (i.name||"").toLowerCase().indexOf("card") >= 0 || (i.placeholder||"").indexOf("カード") >= 0);
    var elB = document.getElementById("birthday") || document.querySelector("input[name='b']") || inps.find(i => (i.name||"").toLowerCase().indexOf("birth") >= 0 || (i.placeholder||"").indexOf("生年月日") >= 0);
    var elP = document.getElementById("password") || document.querySelector("input[type='password'], input[name='p']");

    if (!elC || !elB || !elP) {
      // 順序によるフォールバック (text/tel 2つ + password 1つ)
      var textInps = inps.filter(i => i.type === "text" || i.type === "tel" || i.type === "number");
      var passInps = inps.filter(i => i.type === "password");
      if (textInps.length >= 2 && passInps.length >= 1) {
        elC = textInps[0];
        elB = textInps[1];
        elP = passInps[0];
      }
    }

    if (elC && elB && elP) {
      elC.value = c;
      elB.value = b;
      elP.value = p;
      if (elC.onchange) elC.onchange();
      if (elB.onchange) elB.onchange();
      if (elP.onchange) elP.onchange();

      // ログインボタンの押下
      var btn = Array.from(document.querySelectorAll("a,button,input[type='submit']")).find(el => {
        var t = el.textContent || el.value || "";
        return t.indexOf("ログイン") >= 0 || t.indexOf("認証") >= 0;
      });
      if (btn) {
        btn.click();
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
    } else {
      var regularBtn = Array.from(document.querySelectorAll("a,button")).find(b => {
        var t = b.textContent || "";
        return t.indexOf("通常投票") >= 0;
      });
      if (regularBtn) regularBtn.click();
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
      var x = document.getElementById("umaca-diag");
      if (!x) {
        x = document.createElement("div");
        x.id = "umaca-diag";
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:100000;background:" + (color || "rgba(147,51,234,0.95)") + ";color:#ffffff;font-size:12px;font-weight:bold;padding:8px 12px;pointer-events:none;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.3);";
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

        dg("🟣 UMACA自動選択中... Step " + (i + 1) + "/" + s.length);

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
