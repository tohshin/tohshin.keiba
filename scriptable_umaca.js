// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: purple; icon-glyph: id-card;

/**
 * JRA UMACA スマート ui-title完全一致型 自動投票スクリプト (for Scriptable)
 * 画面上部の class="ui-title" のテキストから現在画面を100%正確に判定して自動進行します。
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

  if (!params) {
    let a = new Alert();
    a.title = "買い目データが見つかりません";
    a.message = "予想サイト上の「UMACA」ボタンから実行してください。";
    a.addAction("OK");
    await a.present();
    Script.complete();
    return;
  }

  const { steps, venueName, placeName, weekday, unitAmount, totalAmount, round, raceNo, siki, hou, axes, partners, isMulti } = params;
  const targetVenue = venueName || placeName || "";
  const targetRace = String(round || raceNo || (steps && steps[1]) || "1").replace(/[^0-9]/g, "");
  const targetSiki = String(siki || (steps && steps[2]) || "1");
  const targetHou = String(hou || (steps && steps[3]) || "0");
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

  // 4. ui-title判定オートメーションの注入
  let automationScript = `
  (function() {
    var cardNo = ${JSON.stringify(cardNo)};
    var birthDay = ${JSON.stringify(birthDay)};
    var passNo = ${JSON.stringify(passNo)};
    var venue = ${JSON.stringify(targetVenue)};
    var raceNum = ${JSON.stringify(targetRace)};
    var sikiCode = ${JSON.stringify(targetSiki)};
    var houCode = ${JSON.stringify(targetHou)};
    var axes = ${JSON.stringify(axes || [])};
    var partners = ${JSON.stringify(partners || [])};
    var rawSteps = ${JSON.stringify(steps || [])};
    var hundreds = ${hundreds};
    var totalAmount = ${tAmount};
    var isMulti = ${Boolean(isMulti)};

    var sikiMap = {
      "1": ["単勝"],
      "2": ["複勝"],
      "3": ["枠連"],
      "4": ["馬連"],
      "5": ["ワイド"],
      "6": ["馬単"],
      "7": ["３連複", "3連複"],
      "8": ["３連単", "3連単"]
    };

    function dg(m, col) {
      var x = document.getElementById("umaca-diag");
      if (!x) {
        x = document.createElement("div");
        x.id = "umaca-diag";
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:100000;background:" + (col || "rgba(147,51,234,0.95)") + ";color:#ffffff;font-size:12px;font-weight:bold;padding:9px 12px;pointer-events:none;text-align:center;box-shadow:0 3px 10px rgba(0,0,0,0.35);line-height:1.4;";
        document.body.appendChild(x);
      }
      x.innerText = m;
      if (col) x.style.background = col;
    }

    function trigger(el) {
      if (!el) return;
      var href = el.getAttribute("href") || "";
      if (href.toLowerCase().startsWith("javascript:")) {
        try {
          var code = href.replace(/^javascript:/i, '');
          window.eval(code);
          return;
        } catch (e) {}
      }
      try {
        var rect = el.getBoundingClientRect();
        var x = rect.left + rect.width / 2;
        var y = rect.top + rect.height / 2;
        var o = {bubbles:true, cancelable:true, clientX:x, clientY:y, view:window};
        el.dispatchEvent(new MouseEvent("mousedown", o));
        el.dispatchEvent(new MouseEvent("mouseup", o));
        el.dispatchEvent(new MouseEvent("click", o));
      } catch (e) {}
      try { el.click(); } catch(e) {}
    }

    var startTime = Date.now();
    var lastAction = "";
    var actionCooldown = 0;

    function runLoop() {
      if (Date.now() - startTime > 45000) {
        dg("⚠️ 処理がタイムアウトしました", "rgba(239,68,68,0.95)");
        return;
      }
      if (Date.now() < actionCooldown) {
        setTimeout(runLoop, 250);
        return;
      }

      try {
        var bodyText = document.body ? (document.body.innerText || "") : "";
        var allLinks = Array.from(document.querySelectorAll("a, button, input[type='button'], input[type='submit']"));

        // 1. class="ui-title" の取得（現在画面の判定基準）
        var titleEl = document.querySelector(".ui-page-active .ui-title, .ui-title, h1.ui-title");
        var title = titleEl ? (titleEl.innerText || titleEl.textContent || "").trim() : "";

        // A. ログイン画面
        var inps = Array.from(document.querySelectorAll("input"));
        var elC = document.getElementById("cardno") || document.querySelector("input[name='c']") || inps.find(i => (i.name||"").toLowerCase().indexOf("card") >= 0 || (i.placeholder||"").indexOf("カード") >= 0);
        var elB = document.getElementById("birthday") || document.querySelector("input[name='b']") || inps.find(i => (i.name||"").toLowerCase().indexOf("birth") >= 0 || (i.placeholder||"").indexOf("生年月日") >= 0);
        var elP = document.getElementById("password") || document.querySelector("input[type='password'], input[name='p']");

        if (!elC || !elB || !elP) {
          var textInps = inps.filter(i => i.type === "text" || i.type === "tel" || i.type === "number");
          var passInps = inps.filter(i => i.type === "password");
          if (textInps.length >= 2 && passInps.length >= 1) {
            elC = textInps[0];
            elB = textInps[1];
            elP = passInps[0];
          }
        }

        if (elC && elB && elP && elC.value !== cardNo) {
          dg("🟣 UMACAへログイン中...");
          elC.value = cardNo;
          elB.value = birthDay;
          elP.value = passNo;
          if (elC.onchange) elC.onchange();
          if (elB.onchange) elB.onchange();
          if (elP.onchange) elP.onchange();

          var btn = allLinks.find(el => {
            var t = el.textContent || el.value || "";
            return t.indexOf("ログイン") >= 0 || t.indexOf("認証") >= 0;
          });
          if (btn) btn.click();
          else {
            var form = document.querySelector("form");
            if (form) form.submit();
          }
          lastAction = "LOGIN";
          actionCooldown = Date.now() + 1000;
          setTimeout(runLoop, 500);
          return;
        }

        // ログインエラー検出
        if (bodyText.indexOf("エラー") >= 0 && (bodyText.indexOf("カード番号") >= 0 || bodyText.indexOf("暗証番号") >= 0)) {
          dg("❌ ログイン情報に誤りがあります", "rgba(239,68,68,0.95)");
          return;
        }

        // B. 最終確認画面
        var passInput = document.querySelector("input[type='password'], #password, input[name='p']");
        var textInputs = Array.from(document.querySelectorAll("input[type='tel'], input[type='text'], input[type='number']"));
        if ((title.indexOf("確認") >= 0 || passInput) && (bodyText.indexOf("合計") >= 0 || bodyText.indexOf("投票内容") >= 0 || bodyText.indexOf("購入") >= 0)) {
          dg("📝 確認画面に暗証番号と金額を入力中...");
          for (var j = 0; j < textInputs.length; j++) {
            var inp = textInputs[j];
            var iid = (inp.id || "").toLowerCase();
            var iname = (inp.name || "").toLowerCase();
            if (iid.indexOf("pass") < 0 && iname.indexOf("pass") < 0) {
              inp.value = String(totalAmount);
              if (inp.onchange) inp.onchange();
              if (inp.oninput) inp.oninput();
              break;
            }
          }
          if (passInput) {
            passInput.value = passNo;
            if (passInput.onchange) passInput.onchange();
            if (passInput.oninput) passInput.oninput();
          }

          dg("✅ セット完了！内容を確認し、よろしければ【投票】を押してください", "rgba(16,185,129,0.95)");
          return;
        }

        // C. 金額入力セット後、確認画面へ進むボタン
        var toSendBtn = allLinks.find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return t === "入力終了" || t === "投票確認" || t.indexOf("入力終了") >= 0 || t.indexOf("投票確認") >= 0;
        });
        if (toSendBtn && (lastAction === "SET_BET" || bodyText.indexOf("投票リスト") >= 0)) {
          dg("📑 投票確認画面へ進みます...");
          if (typeof ToSend === "function") ToSend();
          else trigger(toSendBtn);
          lastAction = "TO_SEND";
          actionCooldown = Date.now() + 1200;
          setTimeout(runLoop, 600);
          return;
        }

        // D. 「金額入力」画面
        if (title.indexOf("金額") >= 0 || document.getElementById("kin") || document.querySelector("#kin .amount input")) {
          var kinInput = document.querySelector("#kin .amount input, input.amount, input[name='amount'], input[type='tel']");
          var setBtn = allLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return t === "セット" || t.indexOf("セット") >= 0;
          });
          if (lastAction !== "SET_BET") {
            dg("💰 金額を入力・セット中 (1点: " + (hundreds * 100) + "円)...");
            if (kinInput) {
              kinInput.value = String(hundreds);
              if (kinInput.onchange) kinInput.onchange();
              if (kinInput.oninput) kinInput.oninput();
            }
            if (typeof VMA === "function") {
              VMA(String(hundreds));
            }
            setTimeout(function() {
              if (typeof SetBet === "function") SetBet(0);
              else if (setBtn) trigger(setBtn);
            }, 300);

            lastAction = "SET_BET";
            actionCooldown = Date.now() + 1200;
            setTimeout(runLoop, 800);
            return;
          }
        }

        // E. 馬番選択画面
        var isHorseTitle = title.indexOf("単勝") >= 0 || title.indexOf("複勝") >= 0 || title.indexOf("ボックス") >= 0 ||
                           title.indexOf("軸") >= 0 || title.indexOf("相手") >= 0 ||
                           title.indexOf("1着") >= 0 || title.indexOf("2着") >= 0 || title.indexOf("3着") >= 0;

        var horseBtns = allLinks.filter(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          var dv = a.getAttribute("data-value") || "";
          var n = parseInt(t) || parseInt(dv);
          return n >= 1 && n <= 18 && (t === String(n) || t === ("0" + n) || dv === String(n));
        });

        if ((isHorseTitle || horseBtns.length >= 5) && lastAction !== "SELECT_HORSE" && lastAction !== "SET_BET") {
          var allHorses = [];
          if (axes && axes.length > 0) allHorses = allHorses.concat(axes);
          if (partners && partners.length > 0) allHorses = allHorses.concat(partners);
          if (allHorses.length === 0 && rawSteps.length > 3) allHorses = rawSteps.slice(3);

          dg("🐎 馬番を選択中 (" + title + "): " + allHorses.join(", "));

          if (title.indexOf("相手") >= 0 && partners && partners.length > 0) {
            partners.forEach(function(h) {
              var hStr = String(parseInt(h));
              var el = horseBtns.find(function(a) {
                var t = (a.innerText || a.textContent || "").trim();
                var dv = a.getAttribute("data-value") || "";
                return t === hStr || dv === hStr;
              });
              if (el) trigger(el);
              else if (typeof SelectHorse === "function") SelectHorse(hStr);
            });
            if (isMulti) {
              var mb = document.querySelector("input[name='multi'], #multi, input[type='checkbox']");
              if (mb && !mb.checked) mb.click();
            }
          } else {
            var targetList = (axes && axes.length > 0) ? axes : allHorses;
            targetList.forEach(function(h) {
              var hStr = String(parseInt(h));
              var el = horseBtns.find(function(a) {
                var t = (a.innerText || a.textContent || "").trim();
                var dv = a.getAttribute("data-value") || "";
                return t === hStr || dv === hStr || t === ("0" + hStr);
              });
              if (el) trigger(el);
              else if (typeof SelectHorse === "function") SelectHorse(hStr);
            });

            if (axes && axes.length > 0 && partners && partners.length > 0) {
              var nextPartnerBtn = allLinks.find(function(a) {
                var t = (a.innerText || a.textContent || "").trim();
                return t.indexOf("相手") >= 0 || t.indexOf("次へ") >= 0;
              });
              if (nextPartnerBtn) {
                trigger(nextPartnerBtn);
                actionCooldown = Date.now() + 600;
                setTimeout(runLoop, 400);
                return;
              }
            }
          }

          lastAction = "SELECT_HORSE";
          actionCooldown = Date.now() + 900;
          setTimeout(runLoop, 600);
          return;
        }

        // F. 「方式」画面
        if (title.indexOf("方式") >= 0 && lastAction !== "SELECT_HOU" && lastAction !== "SELECT_HORSE") {
          var hName = "通常";
          if (houCode === "1" || houCode === "box") hName = "ボックス";
          else if (houCode === "2" || houCode === "nagashi" || houCode === "multi") hName = "ながし";

          dg("📐 方式を選択中: " + hName);
          var matchedHou = allLinks.find(function(a) {
            return (a.innerText || a.textContent || "").trim().indexOf(hName) >= 0;
          });
          if (matchedHou) trigger(matchedHou);

          lastAction = "SELECT_HOU";
          actionCooldown = Date.now() + 800;
          setTimeout(runLoop, 500);
          return;
        }

        // G. 「式別」画面
        if (title.indexOf("式別") >= 0 && lastAction !== "SELECT_SIKI" && lastAction !== "SELECT_HOU" && lastAction !== "SELECT_HORSE") {
          var targets = sikiMap[String(sikiCode)] || ["単勝"];
          dg("🎯 式別を選択中: " + targets[0]);
          var matchedSiki = allLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return targets.some(function(tgt) { return t.indexOf(tgt) >= 0; });
          });
          if (matchedSiki) trigger(matchedSiki);

          lastAction = "SELECT_SIKI";
          actionCooldown = Date.now() + 800;
          setTimeout(runLoop, 500);
          return;
        }

        // H. 「レース」画面
        if (title.indexOf("レース") >= 0 && lastAction !== "SELECT_RACE" && lastAction !== "SELECT_SIKI") {
          dg("🏁 レースを選択中: " + raceNum + "R");
          var matchedRace = allLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return t === raceNum + "R" || t === raceNum + "レース" || t === raceNum;
          });
          if (matchedRace) trigger(matchedRace);

          lastAction = "SELECT_RACE";
          actionCooldown = Date.now() + 800;
          setTimeout(runLoop, 500);
          return;
        }

        // I. 「競馬場名」画面
        if ((title.indexOf("競馬場") >= 0 || title.indexOf("場名") >= 0 || title === "競馬場名") && lastAction !== "SELECT_VENUE" && lastAction !== "SELECT_RACE") {
          dg("🏇 競馬場を選択中: " + (venue || "開催場"));
          var matchedVenue = allLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return venue && t.indexOf(venue) >= 0;
          });
          if (matchedVenue) trigger(matchedVenue);

          lastAction = "SELECT_VENUE";
          actionCooldown = Date.now() + 800;
          setTimeout(runLoop, 500);
          return;
        }

        // J. メニュー画面（トップ画面）
        // ユーザー提供: <img src="tmpl/images/qr_service_logo.png"> はトップ画面
        var isTopLogo = !!document.querySelector("img[src*='qr_service_logo']");
        var regBtn = document.querySelector("a.ico_regular") || allLinks.find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return t === "通常投票" || t.indexOf("通常投票") >= 0;
        });
        if ((isTopLogo || regBtn) && title.indexOf("競馬場") < 0 && title.indexOf("レース") < 0 && title.indexOf("式別") < 0 && title.indexOf("方式") < 0) {
          dg("📋 トップ画面検出！通常投票へ進みます...");
          if (typeof ToSPBet === "function") ToSPBet(0);
          else if (typeof ToQRBet === "function") ToQRBet();
          else if (regBtn) trigger(regBtn);

          lastAction = "MENU_TO_BET";
          actionCooldown = Date.now() + 1000;
          setTimeout(runLoop, 600);
          return;
        }

        setTimeout(runLoop, 350);
      } catch(err) {
        dg("エラー: " + err.message, "rgba(239,68,68,0.95)");
        setTimeout(runLoop, 600);
      }
    }

    setTimeout(runLoop, 600);
  })();
  `;

  await wv.evaluateJavaScript(automationScript, false);

  // 5. WebViewの終了待機
  await presentPromise;
  Script.complete();
}

await main();
