// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: purple; icon-glyph: id-card;

/**
 * JRA UMACA スマート ui-title完全一致型 自動投票スクリプト (for Scriptable)
 * ページ遷移後も外側ループから確実に再注入し、上部バーのリアルタイム表示と自動遷移を継続します。
 * <a class="ico_regular ui-link">通常投票</a> を確実にタップして競馬場名画面へ進みます。
 * 金額入力までいかず5秒停止した場合は、自動的にTOP画面からやり直します。
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

  const { steps, venueName, placeName, weekday, unitAmount, totalAmount, round, raceNo, siki, hou, axes, partners, isMulti, resetLogin } = params;
  const targetVenue = venueName || placeName || "";
  const targetRace = String(round || raceNo || (steps && steps[1]) || "1").replace(/[^0-9]/g, "");
  const targetSiki = String(siki || (steps && steps[2]) || "1");
  const targetHou = String(hou || (steps && steps[3]) || "0");
  const uAmount = parseInt(unitAmount) || 100;
  const hundreds = Math.floor(uAmount / 100);
  const tAmount = parseInt(totalAmount) || uAmount;

  if (resetLogin) {
    if (Keychain.contains("umaca_card_no")) Keychain.remove("umaca_card_no");
    if (Keychain.contains("umaca_birth_day")) Keychain.remove("umaca_birth_day");
    if (Keychain.contains("umaca_pass_no")) Keychain.remove("umaca_pass_no");
  }

  // 2. Keychain から UMACA ログイン情報を取得 (初回またはリセット時はプロンプト)
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

  // 4. ページ遷移に強い継続注入オートメーション
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
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:2147483647;background:" + (col || "rgba(147,51,234,0.95)") + ";color:#ffffff;font-size:12px;font-weight:bold;padding:9px 12px;pointer-events:none;text-align:center;box-shadow:0 3px 10px rgba(0,0,0,0.35);line-height:1.4;";
        if (document.body) document.body.appendChild(x);
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

      // 5. jQuery Mobile 用
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

    // 状態管理用グローバル変数（ページ遷移しても再初期化しつつ保持）
    if (!window._umacaState) {
      window._umacaState = {
        lastStepName: "",
        loginAttemptCount: 0,
        actionCooldown: 0
      };
    }
    var st = window._umacaState;

    if (Date.now() < st.actionCooldown) {
      return { status: "COOLDOWN" };
    }

    try {
      var bodyText = document.body ? (document.body.innerText || "") : "";
      var activePage = document.querySelector(".ui-page-active") || document;
      var allLinks = Array.from(activePage.querySelectorAll("a, button, input[type='button'], input[type='submit'], li.ui-btn, [role='button']"));

      // 1. class="ui-title" の取得（現在アクティブな画面タイトルの判定）
      var titleEl = activePage.querySelector(".ui-title, h1.ui-title") || document.querySelector(".ui-page-active .ui-title, .ui-title");
      var title = titleEl ? (titleEl.innerText || titleEl.textContent || "").trim() : "";

      // 0. お知らせ画面/ポップアップの自動クリア（「今後はこのお知らせを表示しない」にチェックしてOK）
      var forceInfoCb = document.getElementById("force_info_checkbox") ||
                        document.querySelector("input[name='force_info_checkbox'], input[id*='force_info'], input[id*='notice']");
      if (!forceInfoCb) {
        var allLabels = Array.from(document.querySelectorAll("label"));
        var noticeLabel = allLabels.find(function(l) {
          var t = (l.innerText || l.textContent || "").trim();
          return t.indexOf("お知らせを表示しない") >= 0 || t.indexOf("表示しない") >= 0;
        });
        if (noticeLabel) {
          var forId = noticeLabel.getAttribute("for");
          if (forId) forceInfoCb = document.getElementById(forId);
          if (!forceInfoCb) forceInfoCb = noticeLabel.querySelector("input[type='checkbox']");
        }
      }

      if (forceInfoCb || title.indexOf("お知らせ") >= 0 || bodyText.indexOf("今後はこのお知らせを表示しない") >= 0) {
        dg("📢 【お知らせ検知】 「表示しない」にチェックしてOKを押下中...");
        if (forceInfoCb && !forceInfoCb.checked) {
          trigger(forceInfoCb);
        }
        var okBtn = allLinks.find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return t === "OK" || t.indexOf("OK") >= 0 || t === "閉じる";
        }) || Array.from(document.querySelectorAll("a, button, input[type='button']")).find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return t === "OK" || t.indexOf("OK") >= 0 || t === "閉じる";
        });

        if (okBtn) {
          setTimeout(function() { trigger(okBtn); }, 100);
          st.actionCooldown = Date.now() + 1000;
          return { status: "NOTICE_CLEARED" };
        }
      }

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

      if (elC && elB && elP) {
        if (st.loginAttemptCount >= 1) {
          var errMsg = "UMACAログインに失敗しました。";
          if (bodyText.indexOf("エラー") >= 0 || bodyText.indexOf("カード番号") >= 0 || bodyText.indexOf("暗証番号") >= 0) {
            errMsg = "カード番号・生年月日・暗証番号に誤りがあります。";
          }
          dg("⚠️ " + errMsg, "rgba(239,68,68,0.95)");
          return { status: "LOGIN_ERROR" };
        }

        st.loginAttemptCount++;
        dg("🟣 【ログイン】 UMACAへログイン情報を送信中...");
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
        st.actionCooldown = Date.now() + 1500;
        return { status: "LOGIN_SUBMITTED" };
      }

      // ログインエラー検出
      if (bodyText.indexOf("エラー") >= 0 && (bodyText.indexOf("カード番号") >= 0 || bodyText.indexOf("暗証番号") >= 0)) {
        dg("❌ ログイン情報に誤りがあります", "rgba(239,68,68,0.95)");
        return { status: "LOGIN_ERROR" };
      }

      // B. 画面ステップの判定
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

      var houCandidateBtns = allLinks.filter(function(a) {
        var t = (a.innerText || a.textContent || "").trim();
        return t === "通常" || t === "ボックス" || t === "ながし" || t.indexOf("通常") >= 0 || t.indexOf("ボックス") >= 0 || t.indexOf("ながし") >= 0;
      });
      var isHouScreen = (title.indexOf("方式") >= 0 || houCandidateBtns.length >= 2) && !isHorseScreen && !isAmountScreen;

      var allSikiNames = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "３連複", "3連複", "３連単", "3連単"];
      var sikiCandidateBtns = allLinks.filter(function(a) {
        var t = (a.innerText || a.textContent || "").trim();
        return allSikiNames.some(function(s) { return t === s || t.indexOf(s) >= 0; });
      });
      var isSikiScreen = (title.indexOf("式別") >= 0 || sikiCandidateBtns.length >= 3) && !isHouScreen && !isHorseScreen && !isAmountScreen;

      var raceCandidateBtns = allLinks.filter(function(a) {
        var t = (a.innerText || a.textContent || "").replace(/\s+/g, " ").trim();
        return /^[0-9]{1,2}\s*R(\s|$)/i.test(t);
      });
      var isRaceScreen = (title.indexOf("レース") >= 0 || raceCandidateBtns.length >= 3) && !isSikiScreen && !isHouScreen && !isHorseScreen && !isAmountScreen;

      var isVenueScreen = (title.indexOf("競馬場") >= 0 || title.indexOf("場名") >= 0 || title === "競馬場名") && !isRaceScreen;

      // TOP画面判定：<a class="ico_regular ui-link">通常投票</a> がある、または logoHeader がある
      var regBtn = document.querySelector(".ui-page-active a.ico_regular, a.ico_regular.ui-link, a.ico_regular, a[class*='ico_regular']") ||
                   allLinks.find(function(a) {
                     var t = (a.innerText || a.textContent || "").trim();
                     return t === "通常投票" || t.indexOf("通常投票") >= 0;
                   }) ||
                   Array.from(document.querySelectorAll("a")).find(function(a) {
                     var t = (a.innerText || a.textContent || "").trim();
                     return t === "通常投票" || t.indexOf("通常投票") >= 0;
                   });

      var hasLogoHeader = !!document.querySelector(".logoHeader, #logoHeader, [class*='logoHeader'], [id*='logoHeader']");
      var isTopScreen = !isNoticeScreen && (regBtn || hasLogoHeader || typeof ToQRBet === "function" || typeof ToSPBet === "function") && !isVenueScreen && !isRaceScreen && !isSikiScreen && !isHouScreen && !isHorseScreen && !isAmountScreen && !isConfirmScreen;

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

      if (detectedStep !== st.lastStepName && detectedStep !== "WAIT") {
        st.lastStepName = detectedStep;
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

        dg("🎉 【セット完了】 内容を確認し【QR作成】ボタンを押してください！", "rgba(16,185,129,0.95)");
        return { status: "DONE" };
      }

      // 2. 金額入力セット後の確認画面への送信ボタン
      var toSendBtn = allLinks.find(function(a) {
        var t = (a.innerText || a.textContent || "").trim();
        return t === "入力終了" || t === "投票確認" || t.indexOf("入力終了") >= 0 || t.indexOf("投票確認") >= 0;
      });
      if (toSendBtn && (bodyText.indexOf("投票リスト") >= 0 || bodyText.indexOf("セット完了") >= 0)) {
        dg("📑 【投票確認へ】 確認画面へ進みます...");
        if (typeof ToSend === "function") ToSend();
        else trigger(toSendBtn);
        st.actionCooldown = Date.now() + 1200;
        return { status: "TO_SEND" };
      }

      // 3. 金額入力画面
      if (isAmountScreen) {
        var kinInput = activePage.querySelector("#kin .amount input, input.amount, input[name='amount'], input[type='tel']");
        var setBtn = allLinks.find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return t === "セット" || t.indexOf("セット") >= 0;
        });

        // マルチ指定の反映（マルチチェックボックスは金額入力画面 #kin に存在）
        if (isMulti) {
          var mb = activePage.querySelector("#multi, input[name='multi'], input[type='checkbox'][id*='multi']") ||
                   document.querySelector("#multi, input[name='multi'], input[type='checkbox'][id*='multi']");
          if (mb && !mb.checked) {
            trigger(mb);
          }
        }

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
        }, 200);

        st.actionCooldown = Date.now() + 1200;
        return { status: "AMOUNT_SET" };
      }

      // 4. 馬番選択画面
      if (isHorseScreen) {
        var allHorses = [];
        if (axes && axes.length > 0) allHorses = allHorses.concat(axes);
        if (partners && partners.length > 0) allHorses = allHorses.concat(partners);
        if (allHorses.length === 0 && rawSteps.length > 3) allHorses = rawSteps.slice(3);

        function matchHorse(el, h) {
          if (!el) return false;
          var t = (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
          var hStr = String(parseInt(h, 10));
          var p1 = new RegExp("^0*" + hStr + "(?:\\s|$|[^0-9])");
          if (p1.test(t)) return true;
          var dv = el.getAttribute("data-value") || el.getAttribute("data-horse") || el.getAttribute("id") || "";
          if (dv === hStr || dv === ("0" + hStr) || dv === "horse_" + hStr) return true;
          return false;
        }

        dg("🐎 【馬番選択】 " + (title || "馬番") + " を選択中...");

        // A. 相手選択画面（タイトルに「相手」が含まれる場合）
        if (title.indexOf("相手") >= 0) {
          var unselectedPartners = (partners && partners.length > 0) ? partners : allHorses;
          unselectedPartners.forEach(function(h) {
            var el = horseBtns.find(function(a) { return matchHorse(a, h); });
            if (!el) {
              var docLinks = Array.from(document.querySelectorAll("a, button, li.ui-btn"));
              el = docLinks.find(function(a) { return matchHorse(a, h); });
            }
            if (el && !el.classList.contains("selected")) {
              trigger(el);
            } else if (!el && typeof SelectHorse === "function") {
              SelectHorse(String(parseInt(h, 10)));
            }
          });

          // 相手選択完了後、金額入力画面へ遷移するボタンをタップ
          var toKinBtn = allLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            var href = a.getAttribute("href") || "";
            return t.indexOf("金額入力") >= 0 || href.indexOf("#kin") >= 0;
          }) || Array.from(document.querySelectorAll("a, button")).find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            var href = a.getAttribute("href") || "";
            return t.indexOf("金額入力") >= 0 || href.indexOf("#kin") >= 0;
          });

          if (toKinBtn) {
            trigger(toKinBtn);
            st.actionCooldown = Date.now() + 800;
            return { status: "TO_KIN" };
          }

          st.actionCooldown = Date.now() + 800;
          return { status: "PARTNERS_SELECTED" };
        } else {
          // B. 軸馬または通常/ボックスの馬番選択（1着軸画面など）
          var targetList = (axes && axes.length > 0) ? axes : allHorses;
          targetList.forEach(function(h) {
            var el = horseBtns.find(function(a) { return matchHorse(a, h); });
            if (!el) {
              var docLinks = Array.from(document.querySelectorAll("a, button, li.ui-btn"));
              el = docLinks.find(function(a) { return matchHorse(a, h); });
            }
            if (el) {
              trigger(el);
            } else if (typeof SelectHorse === "function") {
              SelectHorse(String(parseInt(h, 10)));
            }
          });

          // ボックスまたは通常で「金額入力画面へ」がある場合
          var toKinBtn2 = allLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            var href = a.getAttribute("href") || "";
            return t.indexOf("金額入力") >= 0 || href.indexOf("#kin") >= 0;
          });
          if (toKinBtn2) {
            trigger(toKinBtn2);
            st.actionCooldown = Date.now() + 800;
            return { status: "TO_KIN" };
          }

          st.actionCooldown = Date.now() + 800;
          return { status: "HORSE_CLICKED" };
        }
      }

      // 5. 方式選択画面
      if (isHouScreen) {
        var isBox = (houCode === "1" || houCode === "box" || houCode === "2");
        var isNagashi = (!isBox && partners && partners.length > 0) || 
                        houCode === "nagashi" || houCode === "multi" || 
                        houCode === "3" || houCode === "6" || houCode === "7";

        var targetLabels = [];
        if (isBox) {
          targetLabels = ["ボックス", "BOX"];
        } else if (isNagashi) {
          var numAxes = (axes && axes.length) ? axes.length : 1;
          if (sikiCode === "8") { // 3連単
            if (numAxes >= 2) {
              targetLabels = ["1･2着ながし", "1・2着ながし", "軸2頭ながし", "ながし"];
            } else {
              targetLabels = ["1着ながし", "軸1頭ながし", "ながし"];
            }
          } else if (sikiCode === "7") { // 3連複
            if (numAxes >= 2) {
              targetLabels = ["軸2頭ながし", "1･2着ながし", "ながし"];
            } else {
              targetLabels = ["軸1頭ながし", "1着ながし", "ながし"];
            }
          } else if (sikiCode === "6") { // 馬単
            targetLabels = ["1着ながし", "ながし"];
          } else { // 馬連、ワイド、枠連
            targetLabels = ["ながし", "軸1頭ながし"];
          }
        } else {
          targetLabels = ["通常"];
        }

        dg("📐 【方式選択】 「" + targetLabels[0] + "」を選択中...");

        function matchHou(el) {
          if (!el) return false;
          var t = (el.innerText || el.textContent || "").trim();
          return targetLabels.some(function(lbl) {
            return t === lbl || t.indexOf(lbl) >= 0;
          });
        }

        var matchedHou = allLinks.find(matchHou);
        if (!matchedHou) {
          var docLinks = Array.from(document.querySelectorAll("a, button, li.ui-btn, [role='button']"));
          matchedHou = docLinks.find(matchHou);
        }
        if (matchedHou) {
          trigger(matchedHou);
          st.actionCooldown = Date.now() + 800;
          return { status: "HOU_CLICKED" };
        } else {
          dg("📐 【方式選択】 「" + targetLabels[0] + "」を探しています...");
          return { status: "HOU_SEARCHING" };
        }
      }

      // 6. 式別選択画面
      if (isSikiScreen) {
        var targets = sikiMap[String(sikiCode)] || ["単勝"];
        dg("🎯 【式別選択】 「" + targets[0] + "」を選択中...");
        var matchedSiki = allLinks.find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return targets.some(function(tgt) { return t.indexOf(tgt) >= 0; });
        });
        if (!matchedSiki) {
          var docLinks = Array.from(document.querySelectorAll("a, button, li.ui-btn, [role='button']"));
          matchedSiki = docLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return targets.some(function(tgt) { return t.indexOf(tgt) >= 0; });
          });
        }
        if (matchedSiki) {
          trigger(matchedSiki);
          st.actionCooldown = Date.now() + 800;
          return { status: "SIKI_CLICKED" };
        }
      }

      // 7. レース選択画面
      if (isRaceScreen) {
        var targetRStr = String(parseInt(raceNum, 10));
        dg("🏁 【レース選択】 " + targetRStr + "R を選択中...");

        function matchRace(el) {
          if (!el) return false;
          var t = (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
          var r1 = new RegExp("^0*" + targetRStr + "\\s*R(?:\\s|$|[^0-9])", "i");
          if (r1.test(t)) return true;
          var r2 = new RegExp("^0*" + targetRStr + "\\s*レース(?:\\s|$)", "i");
          if (r2.test(t)) return true;
          if (t === targetRStr || t === ("0" + targetRStr)) return true;
          var dv = el.getAttribute("data-value") || el.getAttribute("data-race") || el.getAttribute("id") || "";
          if (dv === targetRStr || dv === ("0" + targetRStr) || dv === "race_" + targetRStr) return true;
          return false;
        }

        var matchedRace = allLinks.find(matchRace);
        if (!matchedRace) {
          var docLinks = Array.from(document.querySelectorAll("a, button, li.ui-btn, [role='button']"));
          matchedRace = docLinks.find(matchRace);
        }

        if (matchedRace) {
          trigger(matchedRace);
          st.actionCooldown = Date.now() + 800;
          return { status: "RACE_CLICKED" };
        } else {
          dg("🏁 【レース選択】 " + targetRStr + "R のボタンを探しています...");
          return { status: "RACE_SEARCHING" };
        }
      }

      // 8. 競馬場名選択画面
      if (isVenueScreen) {
        dg("🏇 【競馬場選択】 「" + (venue || "開催場") + "」を選択中...");
        var matchedVenue = allLinks.find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return venue && t.indexOf(venue) >= 0;
        });
        if (!matchedVenue) {
          var docLinks = Array.from(document.querySelectorAll("a, button, li.ui-btn, [role='button']"));
          matchedVenue = docLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return venue && t.indexOf(venue) >= 0;
          });
        }
        if (matchedVenue) trigger(matchedVenue);

        st.actionCooldown = Date.now() + 800;
        return { status: "VENUE_CLICKED" };
      }

      // 9. TOP画面（通常投票へ遷移）
      if (isTopScreen) {
        dg("📋 【TOP画面】 通常投票へ進みます...");
        if (typeof ToQRBet === "function") {
          try { ToQRBet(); } catch(e) {}
        } else if (typeof ToSPBet === "function") {
          try { ToSPBet(0); } catch(e) {}
        } else if (regBtn) {
          trigger(regBtn);
        }

        st.actionCooldown = Date.now() + 1200;
        return { status: "TOP_CLICKED" };
      }

      // 10. 読み込み待ち状態の表示
      dg("⏳ 【画面読み込み中】 " + (title ? "タイトル: " + title : "処理待機中..."));
      return { status: "WAITING" };
    } catch(err) {
      dg("⚠️ エラー: " + err.message, "rgba(239,68,68,0.95)");
      return { status: "ERROR", error: err.message };
    }
  })();
  `;

  // 5. 外側（Scriptable側）から定期注入・実行監視ループ（ページ遷移で絶対に死なない）
  let isDone = false;
  let loopStart = Date.now();

  while (!isDone && (Date.now() - loopStart < 60000)) {
    try {
      let res = await wv.evaluateJavaScript(automationScript, false);
      if (res && res.status === "DONE") {
        isDone = true;
        break;
      }
    } catch(e) {
      // 画面遷移中（ページロード中）は一時的に例外になる場合があるが継続
    }
    await new Promise(resolve => Timer.schedule(350, false, resolve));
  }

  // 6. WebViewの終了待機
  await presentPromise;
  Script.complete();
}

await main();
