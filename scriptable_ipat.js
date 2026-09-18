// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: blue; icon-glyph: ticket-alt;

/**
 * JRA 即PAT (IPAT) ui-title完全一致型 自動投票スクリプト (for Scriptable)
 * 画面上部の class="ui-title" のテキストから現在画面を100%正確に判定して自動進行します。
 * 金額入力までいかず5秒停止した場合は、自動的にTOP画面からやり直します。
 * 上部バーに現在の処理内容と経過秒数をリアルタイム表示します。
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
    a.message = "予想サイト上の「即PAT」ボタンから実行してください。";
    a.addAction("OK");
    await a.present();
    Script.complete();
    return;
  }

  const { steps, venueName, placeName, weekday, unitAmount, totalAmount, round, raceNo, siki, hou, axes, partners, isMulti, resetLogin } = params;
  const targetVenue = venueName || placeName || "";
  const targetRace = String(round || raceNo || (steps && steps[1]) || "1").replace(/[^0-9]/g, "");
  const targetSiki = String(siki || (steps && steps[2]) || "1");
  const targetHou = String(hou || (steps && steps[3]) || "0");
  const uAmount = parseInt(unitAmount) || 100;
  const hundreds = Math.floor(uAmount / 100);
  const tAmount = parseInt(totalAmount) || uAmount;

  if (resetLogin) {
    if (Keychain.contains("ipat_user_no")) Keychain.remove("ipat_user_no");
    if (Keychain.contains("ipat_pass_no")) Keychain.remove("ipat_pass_no");
    if (Keychain.contains("ipat_pars_no")) Keychain.remove("ipat_pars_no");
  }

  // 2. Keychain から IPAT ログイン情報を取得 (初回またはリセット時はプロンプト)
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

  // 4. ui-title判定 & 5秒リトライ型オートメーションの注入
  let automationScript = `
  (function() {
    var userNo = ${JSON.stringify(userNo)};
    var passNo = ${JSON.stringify(passNo)};
    var parsNo = ${JSON.stringify(parsNo)};
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
      var x = document.getElementById("ipat-diag");
      if (!x) {
        x = document.createElement("div");
        x.id = "ipat-diag";
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:100000;background:" + (col || "rgba(79,70,229,0.95)") + ";color:#ffffff;font-size:12px;font-weight:bold;padding:9px 12px;pointer-events:none;text-align:center;box-shadow:0 3px 10px rgba(0,0,0,0.35);line-height:1.4;";
        document.body.appendChild(x);
      }
      x.innerText = m;
      if (col) x.style.background = col;
    }

    function trigger(el) {
      if (!el) return;
      try { el.focus(); } catch(e) {}

      // 1. TouchEvent（スマホ/WebViewのタップ）
      try {
        var r = el.getBoundingClientRect();
        var x = r.left + r.width / 2;
        var y = r.top + r.height / 2;
        var t = new Touch({identifier: Date.now(), target: el, clientX: x, clientY: y, radiusX: 2, radiusY: 2});
        var to = {bubbles: true, cancelable: true, touches: [t], targetTouches: [t], changedTouches: [t], view: window};
        el.dispatchEvent(new TouchEvent("touchstart", to));
        el.dispatchEvent(new TouchEvent("touchend", to));
      } catch (err) {}

      // 2. PointerEvent
      try {
        var r2 = el.getBoundingClientRect();
        var x2 = r2.left + r2.width / 2;
        var y2 = r2.top + r2.height / 2;
        el.dispatchEvent(new PointerEvent("pointerdown", {bubbles: true, cancelable: true, clientX: x2, clientY: y2, view: window}));
        el.dispatchEvent(new PointerEvent("pointerup", {bubbles: true, cancelable: true, clientX: x2, clientY: y2, view: window}));
      } catch (err) {}

      // 3. MouseEvent
      try {
        var r3 = el.getBoundingClientRect();
        var x3 = r3.left + r3.width / 2;
        var y3 = r3.top + r3.height / 2;
        var o = {bubbles: true, cancelable: true, clientX: x3, clientY: y3, view: window};
        el.dispatchEvent(new MouseEvent("mousedown", o));
        el.dispatchEvent(new MouseEvent("mouseup", o));
        el.dispatchEvent(new MouseEvent("click", o));
      } catch (err) {}

      // 4. 標準 el.click()
      try { el.click(); } catch (err) {}

      // 5. jQuery Mobile 用 (もしページ内に $ や jQuery がある場合)
      try {
        if (window.$ && typeof window.$(el).trigger === "function") {
          window.$(el).trigger("vclick");
          window.$(el).trigger("tap");
          window.$(el).trigger("click");
        }
      } catch (err) {}

      // 6. onclick / href 属性の直接実行
      var onclick = el.getAttribute("onclick") || "";
      if (onclick) {
        try { window.eval(onclick); } catch (err) {}
      }
      var href = el.getAttribute("href") || "";
      if (href) {
        if (href.toLowerCase().startsWith("javascript:")) {
          try {
            var code = href.replace(/^javascript:/i, '');
            window.eval(code);
          } catch (err) {}
        }
      }
    }

    var startTime = Date.now();
    var lastAction = "";
    var actionCooldown = 0;
    var loginAttemptCount = 0;

    // 5秒停止検知 & TOPやり直し用の状態管理
    var currentStepName = "";
    var lastStepName = "";
    var stepStartTime = Date.now();
    var retryCount = 0;

    function runLoop() {
      if (Date.now() - startTime > 50000) {
        dg("⚠️ 処理がタイムアウトしました", "rgba(239,68,68,0.95)");
        return;
      }
      if (Date.now() < actionCooldown) {
        setTimeout(runLoop, 250);
        return;
      }

      try {
        var bodyText = document.body ? (document.body.innerText || "") : "";
        var activePage = document.querySelector(".ui-page-active") || document;
        var allLinks = Array.from(activePage.querySelectorAll("a, button, input[type='button'], input[type='submit']"));

        // 1. class="ui-title" の取得（現在アクティブな画面タイトルの判定）
        var titleEl = activePage.querySelector(".ui-title, h1.ui-title") || document.querySelector(".ui-page-active .ui-title, .ui-title");
        var title = titleEl ? (titleEl.innerText || titleEl.textContent || "").trim() : "";

        // A. ログイン画面
        var elU = document.getElementById("userid") || document.querySelector("input[name='i']");
        var elP = document.getElementById("password") || document.querySelector("input[name='p']");
        var elR = document.getElementById("pars") || document.querySelector("input[name='r']");
        if (elU && elP && elR) {
          if (loginAttemptCount >= 1) {
            var errMsg = "ログインできませんでした。";
            if (bodyText.indexOf("エラー") >= 0 || bodyText.indexOf("誤り") >= 0 || bodyText.indexOf("022/1010") >= 0) {
              errMsg = "加入者番号・暗証番号・P-ARS番号に誤りがあります。";
            } else if (bodyText.indexOf("時間外") >= 0 || bodyText.indexOf("休止") >= 0 || bodyText.indexOf("メンテナンス") >= 0) {
              errMsg = "現在JRAのサービス提供時間外（メンテナンス中）です。";
            }
            dg("⚠️ " + errMsg, "rgba(239,68,68,0.95)");
            return;
          }

          loginAttemptCount++;
          dg("🔐 【ログイン】 即PATへログイン情報を送信中...");
          elU.value = userNo;
          elP.value = passNo;
          elR.value = parsNo;
          if (elU.onchange) elU.onchange();
          if (elP.onchange) elP.onchange();
          if (elR.onchange) elR.onchange();
          if (typeof ToSPMenu === "function") {
            ToSPMenu();
          } else {
            var form = document.querySelector("form");
            if (form) form.submit();
          }
          lastAction = "LOGIN";
          actionCooldown = Date.now() + 1500;
          setTimeout(runLoop, 800);
          return;
        }

        // ログインエラー検出
        if (bodyText.indexOf("エラー") >= 0 && (bodyText.indexOf("加入者番号") >= 0 || bodyText.indexOf("暗証番号") >= 0 || bodyText.indexOf("022/1010") >= 0)) {
          dg("❌ ログイン情報に誤りがあります", "rgba(239,68,68,0.95)");
          return;
        }

        // B. 現在の画面ステップを特定（判定順序：ゴールである確認・金額画面 -> 各選択画面 -> TOP画面）
        var passInput = activePage.querySelector("input[type='password'], #password, input[name='p']");
        var textInputs = Array.from(activePage.querySelectorAll("input[type='tel'], input[type='text'], input[type='number']"));
        var isConfirmScreen = (title.indexOf("確認") >= 0 || passInput) && (bodyText.indexOf("合計") >= 0 || bodyText.indexOf("投票内容") >= 0 || bodyText.indexOf("購入") >= 0);

        var isAmountScreen = (title.indexOf("金額") >= 0 || document.getElementById("kin") || activePage.querySelector("#kin .amount input"));

        var isHorseTitle = title.indexOf("単勝") >= 0 || title.indexOf("複勝") >= 0 || title.indexOf("ボックス") >= 0 ||
                           title.indexOf("軸") >= 0 || title.indexOf("相手") >= 0 ||
                           title.indexOf("1着") >= 0 || title.indexOf("2着") >= 0 || title.indexOf("3着") >= 0;

        var horseBtns = allLinks.filter(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          var dv = a.getAttribute("data-value") || "";
          var n = parseInt(t) || parseInt(dv);
          return n >= 1 && n <= 18 && (t === String(n) || t === ("0" + n) || dv === String(n));
        });
        var isHorseScreen = (isHorseTitle || horseBtns.length >= 5) && !isAmountScreen && !isConfirmScreen;

        var isHouScreen = (title.indexOf("方式") >= 0) && !isHorseScreen && !isAmountScreen;
        var isSikiScreen = (title.indexOf("式別") >= 0) && !isHouScreen && !isHorseScreen;
        var isRaceScreen = (title.indexOf("レース") >= 0) && !isSikiScreen && !isHouScreen;
        var isVenueScreen = (title.indexOf("競馬場") >= 0 || title.indexOf("場名") >= 0 || title === "競馬場名") && !isRaceScreen;

        // TOP画面判定：アクティブページ内にlogoHeaderがある、または他画面ではなく通常投票ボタンがある
        var hasLogoHeader = !!activePage.querySelector(".logoHeader, #logoHeader, [class*='logoHeader'], [id*='logoHeader']");
        var regBtn = document.querySelector(".ui-page-active a.ico_regular, a.ico_regular.ui-link, a.ico_regular, a[class*='ico_regular']") ||
                     allLinks.find(function(a) {
                       var t = (a.innerText || a.textContent || "").trim();
                       return t === "通常投票" || t.indexOf("通常投票") >= 0;
                     }) ||
                     Array.from(document.querySelectorAll("a")).find(function(a) {
                       var t = (a.innerText || a.textContent || "").trim();
                       return t === "通常投票" || t.indexOf("通常投票") >= 0;
                     });
        var isTopScreen = (hasLogoHeader || regBtn) && !isVenueScreen && !isRaceScreen && !isSikiScreen && !isHouScreen && !isHorseScreen && !isAmountScreen && !isConfirmScreen;

        // ステップ名の決定
        var detectedStep = "";
        if (isConfirmScreen) detectedStep = "CONFIRM";
        else if (isAmountScreen) detectedStep = "AMOUNT";
        else if (isHorseScreen) detectedStep = "HORSE";
        else if (isHouScreen) detectedStep = "HOU";
        else if (isSikiScreen) detectedStep = "SIKI";
        else if (isRaceScreen) detectedStep = "RACE";
        else if (isVenueScreen) detectedStep = "VENUE";
        else if (isTopScreen) detectedStep = "TOP";
        else detectedStep = "WAIT";

        // ステップ変化の監視（同じ画面で5秒止まったらTOPからやり直し）
        if (detectedStep !== lastStepName && detectedStep !== "WAIT") {
          lastStepName = detectedStep;
          stepStartTime = Date.now();
        }

        var stepElapsedSec = Math.floor((Date.now() - stepStartTime) / 1000);

        // 【5秒ルール】金額入力まで行かず、どこかの処理で5秒止まったらTOPからやり直す
        if (detectedStep !== "AMOUNT" && detectedStep !== "CONFIRM" && detectedStep !== "WAIT") {
          if (stepElapsedSec >= 5) {
            retryCount++;
            dg("⚠️ 5秒停止検知: TOP画面からやり直します (" + retryCount + "回目)...", "rgba(239,68,68,0.95)");
            stepStartTime = Date.now();
            lastStepName = "RETRY";

            // TOP画面への復帰処理
            if (typeof ToSPMenu === "function") {
              try { ToSPMenu(); } catch(e) {}
            } else {
              var homeBtn = activePage.querySelector("a[data-rel='back'], a.ico_menu, a[href*='menu'], a.ui-btn-left");
              if (homeBtn) trigger(homeBtn);
              else if (typeof ToSPBet === "function") ToSPBet(0);
            }
            actionCooldown = Date.now() + 1500;
            setTimeout(runLoop, 800);
            return;
          }
        }

        // --- 各画面の自動実行処理 ---

        // 1. 最終確認画面
        if (isConfirmScreen) {
          dg("✅ 【最終確認】 暗証番号と金額を入力中...");
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

          dg("🎉 【セット完了】 内容を確認し【投票】ボタンを押してください！", "rgba(16,185,129,0.95)");
          return; // 全自動処理完了
        }

        // 2. 金額入力セット後の確認画面への送信ボタン
        var toSendBtn = allLinks.find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return t === "入力終了" || t === "投票確認" || t.indexOf("入力終了") >= 0 || t.indexOf("投票確認") >= 0;
        });
        if (toSendBtn && (lastAction === "SET_BET" || bodyText.indexOf("投票リスト") >= 0)) {
          dg("📑 【投票確認へ】 確認画面へ進みます...");
          if (typeof ToSend === "function") ToSend();
          else trigger(toSendBtn);
          lastAction = "TO_SEND";
          actionCooldown = Date.now() + 1200;
          setTimeout(runLoop, 600);
          return;
        }

        // 3. 金額入力画面
        if (isAmountScreen) {
          var kinInput = activePage.querySelector("#kin .amount input, input.amount, input[name='amount'], input[type='tel']");
          var setBtn = allLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return t === "セット" || t.indexOf("セット") >= 0;
          });
          if (lastAction !== "SET_BET") {
            dg("💰 【金額入力】 1点 " + (hundreds * 100) + "円をセット中...");
            if (kinInput) {
              kinInput.value = String(hundreds);
              if (kinInput.onchange) kinInput.onchange();
              if (kinInput.oninput) kinInput.oninput();
            }
            if (typeof VMA === "function") {
              try { VMA(String(hundreds)); } catch(e) {}
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

        // 4. 馬番選択画面
        if (isHorseScreen && lastAction !== "SELECT_HORSE" && lastAction !== "SET_BET") {
          var allHorses = [];
          if (axes && axes.length > 0) allHorses = allHorses.concat(axes);
          if (partners && partners.length > 0) allHorses = allHorses.concat(partners);
          if (allHorses.length === 0 && rawSteps.length > 3) allHorses = rawSteps.slice(3);

          dg("🐎 【馬番選択】 " + title + " (" + allHorses.join(", ") + "番) を選択中... (" + stepElapsedSec + "秒/5秒)");

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
              var mb = activePage.querySelector("input[name='multi'], #multi, input[type='checkbox']");
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

        // 5. 方式選択画面
        if (isHouScreen && lastAction !== "SELECT_HOU" && lastAction !== "SELECT_HORSE") {
          var hName = "通常";
          if (houCode === "1" || houCode === "box") hName = "ボックス";
          else if (houCode === "2" || houCode === "nagashi" || houCode === "multi") hName = "ながし";

          dg("📐 【方式選択】 「" + hName + "」を選択中... (" + stepElapsedSec + "秒/5秒)");
          var matchedHou = allLinks.find(function(a) {
            return (a.innerText || a.textContent || "").trim().indexOf(hName) >= 0;
          });
          if (matchedHou) trigger(matchedHou);

          lastAction = "SELECT_HOU";
          actionCooldown = Date.now() + 800;
          setTimeout(runLoop, 500);
          return;
        }

        // 6. 式別選択画面
        if (isSikiScreen && lastAction !== "SELECT_SIKI" && lastAction !== "SELECT_HOU") {
          var targets = sikiMap[String(sikiCode)] || ["単勝"];
          dg("🎯 【式別選択】 「" + targets[0] + "」を選択中... (" + stepElapsedSec + "秒/5秒)");
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

        // 7. レース選択画面
        if (isRaceScreen && lastAction !== "SELECT_RACE" && lastAction !== "SELECT_SIKI") {
          dg("🏁 【レース選択】 " + raceNum + "Rを選択中... (" + stepElapsedSec + "秒/5秒)");
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

        // 8. 競馬場名選択画面
        if (isVenueScreen && lastAction !== "SELECT_VENUE" && lastAction !== "SELECT_RACE") {
          dg("🏇 【競馬場選択】 「" + (venue || "開催場") + "」を選択中... (" + stepElapsedSec + "秒/5秒)");
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

        // 9. TOP画面（<a class="ico_regular ui-link">通常投票</a> をクリックして競馬場名へ遷移）
        if (isTopScreen) {
          dg("📋 【TOP画面】 通常投票 (<a class='ico_regular ui-link'>) をタップ中... (" + stepElapsedSec + "秒/5秒)");
          if (regBtn) {
            trigger(regBtn);
          } else {
            if (typeof ToSPBet === "function") {
              try { ToSPBet(0); } catch(e) {}
            }
          }

          lastAction = "MENU_TO_BET";
          actionCooldown = Date.now() + 1000;
          setTimeout(runLoop, 600);
          return;
        }

        // 10. 読み込み待ち状態の表示
        dg("⏳ 【画面読み込み中】 " + (title ? "タイトル: " + title : "処理待機中") + " (" + stepElapsedSec + "秒/5秒)");
        setTimeout(runLoop, 350);
      } catch(err) {
        dg("⚠️ エラー: " + err.message, "rgba(239,68,68,0.95)");
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
