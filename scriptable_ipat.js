// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: blue; icon-glyph: ticket-alt;

/**
 * JRA 即PAT (IPAT) 自動投票スクリプト (for Scriptable)
 * 複数買い目の一括連続入力・確認画面（合計金額入力）完全自動対応版
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

  // 入力JSONの正規化
  let bets = [];
  if (params) {
    if (params.bets && Array.isArray(params.bets)) {
      bets = params.bets;
    } else if (params.steps || params.venueName || params.placeName) {
      bets = [params];
    }
  }

  if (bets.length === 0) {
    let a = new Alert();
    a.title = "買い目データが見つかりません";
    a.message = "予想サイト上の「即PAT」ボタンから実行してください。";
    a.addAction("OK");
    await a.present();
    Script.complete();
    return;
  }

  const resetLogin = params.resetLogin || (bets[0] && bets[0].resetLogin);
  const grandTotalAmount = params.totalAmount || bets.reduce((sum, b) => {
    let tot = parseInt(b.totalAmount) || parseInt(b.unitAmount) || 100;
    return sum + tot;
  }, 0);

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

  // 4. オートメーションスクリプト
  let automationScript = `
  (function() {
    var userNo = ${JSON.stringify(userNo)};
    var passNo = ${JSON.stringify(passNo)};
    var parsNo = ${JSON.stringify(parsNo)};
    var bets = ${JSON.stringify(bets)};
    var grandTotal = ${grandTotalAmount};

    if (!window._ipatState) {
      window._ipatState = {
        betIndex: 0,
        lastStepName: "",
        loginAttemptCount: 0,
        actionCooldown: 0,
        selectedPartnerSet: {}
      };
    }
    var st = window._ipatState;

    if (st.betIndex >= bets.length) {
      st.betIndex = bets.length - 1;
    }

    var curBet = bets[st.betIndex] || bets[0];
    var venue = curBet.venueName || curBet.placeName || "";
    var weekday = curBet.weekday || "";
    var raceNum = String(curBet.round || curBet.raceNo || (curBet.steps && curBet.steps[1]) || "1").replace(/[^0-9]/g, "");
    var sikiCode = String(curBet.siki || (curBet.steps && curBet.steps[2]) || "1");
    var houCode = String(curBet.hou || (curBet.steps && curBet.steps[3]) || "0");
    var axes = (curBet.axes || []).map(function(x) { return String(parseInt(x, 10)); });
    var partners = (curBet.partners || []).map(function(x) { return String(parseInt(x, 10)); });
    var rawSteps = curBet.steps || [];
    var uAmount = parseInt(curBet.unitAmount) || 100;
    var hundreds = Math.max(1, Math.floor(uAmount / 100));
    var isMulti = Boolean(curBet.isMulti);

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
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:2147483647;background:" + (col || "rgba(79,70,229,0.95)") + ";color:#ffffff;font-size:12px;font-weight:bold;padding:9px 12px;pointer-events:none;text-align:center;box-shadow:0 3px 10px rgba(0,0,0,0.35);line-height:1.4;";
        if (document.body) document.body.appendChild(x);
      }
      var k = (st.betIndex || 0) + 1;
      var n = bets.length;
      x.innerText = "[" + k + "/" + n + "件目] " + m;
      if (col) x.style.background = col;
    }

    function trigger(el) {
      if (!el) return;
      try { el.focus(); } catch(e) {}

      // 1. TouchEvent
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

      // 3. MouseEvent & Click
      try {
        var r3 = el.getBoundingClientRect();
        var x3 = r3.left + r3.width / 2;
        var y3 = r3.top + r3.height / 2;
        var o = {bubbles: true, cancelable: true, clientX: x3, clientY: y3, view: window};
        el.dispatchEvent(new MouseEvent("mousedown", o));
        el.dispatchEvent(new MouseEvent("mouseup", o));
        el.dispatchEvent(new MouseEvent("click", o));
      } catch (err) {}

      try { el.click(); } catch (err) {}

      // 4. jQuery Mobile 用トリガー
      try {
        if (window.$ && typeof window.$(el).trigger === "function") {
          window.$(el).trigger("vclick");
          window.$(el).trigger("tap");
          window.$(el).trigger("click");
        }
      } catch (err) {}

      // 5. onclick / href
      var onclick = el.getAttribute("onclick") || "";
      if (onclick) {
        try { window.eval(onclick); } catch (err) {}
      }
      var href = el.getAttribute("href") || "";
      if (href && href.toLowerCase().startsWith("javascript:")) {
        try {
          var code = href.replace(/^javascript:/i, '');
          window.eval(code);
        } catch (err) {}
      }
    }

    if (Date.now() < st.actionCooldown) {
      return { status: "COOLDOWN" };
    }

    try {
      var bodyText = document.body ? (document.body.innerText || "") : "";
      var activePage = document.querySelector(".ui-page-active") || document;
      var curHref = location.href || "";
      var curHash = location.hash || "";

      // 0-A. ご注意モーダル（「このまま進む」#GoVote）の自動クリア
      var goVoteBtn = document.getElementById("GoVote") ||
                      document.querySelector("a#GoVote, a[id*='GoVote']") ||
                      Array.from(document.querySelectorAll("a, button")).find(function(a) {
                        return (a.innerText || a.textContent || "").indexOf("このまま進む") >= 0;
                      });
      if (goVoteBtn) {
        dg("⚠️ 【ご注意モーダル検知】 「このまま進む」を押下中...");
        trigger(goVoteBtn);
        st.actionCooldown = Date.now() + 1000;
        return { status: "GO_VOTE_CLEARED" };
      }

      // 0-B. お知らせ画面の自動クリア（「今後はこのお知らせを表示しない」にチェックしてOK）
      var forceInfoCb = document.getElementById("force_info_checkbox") ||
                        document.querySelector("input[name='force_info_checkbox'], input[id*='force_info'], input[id*='notice']");
      if (forceInfoCb || bodyText.indexOf("今後はこのお知らせを表示しない") >= 0) {
        dg("📢 【お知らせ検知】 「表示しない」にチェックしてOKを押下中...");
        if (forceInfoCb && !forceInfoCb.checked) {
          trigger(forceInfoCb);
        }
        var okBtn = Array.from(document.querySelectorAll("a, button, input[type='button']")).find(function(a) {
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
      var elU = document.getElementById("userid") || document.querySelector("input[name='i']");
      var elP = document.getElementById("password") || document.querySelector("input[name='p']");
      var elR = document.getElementById("pars") || document.querySelector("input[name='r']");
      if (elU && elP && elR) {
        if (st.loginAttemptCount >= 1) {
          var errMsg = "ログインできませんでした。";
          if (bodyText.indexOf("エラー") >= 0 || bodyText.indexOf("誤り") >= 0 || bodyText.indexOf("022/1010") >= 0) {
            errMsg = "加入者番号・暗証番号・P-ARS番号に誤りがあります。";
          } else if (bodyText.indexOf("時間外") >= 0 || bodyText.indexOf("休止") >= 0 || bodyText.indexOf("メンテナンス") >= 0) {
            errMsg = "現在JRAのサービス提供時間外（メンテナンス中）です。";
          }
          dg("⚠️ " + errMsg, "rgba(239,68,68,0.95)");
          return { status: "LOGIN_ERROR" };
        }

        st.loginAttemptCount++;
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
        st.actionCooldown = Date.now() + 1500;
        return { status: "LOGIN_SUBMITTED" };
      }

      // 全リンク・ボタンの取得
      var allLinks = Array.from(activePage.querySelectorAll("a, button, input[type='button'], input[type='submit'], li.ui-btn, [role='button']"));
      var titleEl = activePage.querySelector(".ui-title, h1.ui-title") || document.querySelector(".ui-page-active .ui-title, .ui-title");
      var title = titleEl ? (titleEl.innerText || titleEl.textContent || "").trim() : "";

      // B. 画面ステップの判定

      // 1. 最終確認画面 (pw_741_i.cgi / 合計金額入力)
      var sumInput = document.getElementById("sum") || document.querySelector("input#sum, input[name='sum']");
      var isConfirmScreen = (curHref.indexOf("pw_741_i.cgi") >= 0 || title.indexOf("合計金額入力") >= 0 || !!sumInput);

      if (isConfirmScreen) {
        dg("✅ 【最終確認】 合計金額を入力中...");
        if (sumInput) {
          sumInput.value = String(grandTotal);
          if (sumInput.onchange) sumInput.onchange();
          if (sumInput.oninput) sumInput.oninput();
        }
        dg("🎉 【全" + bets.length + "件セット完了】 合計金額(" + grandTotal.toLocaleString() + "円)を入力しました。内容を確認し【投票】ボタンを押してください！", "rgba(16,185,129,0.95)");
        return { status: "DONE" };
      }

      // 2. 投票一覧画面 (#toui / pw_740_i.cgi?w=1#toui)
      var isVoteListScreen = (curHash === "#toui" || (activePage && activePage.id === "toui") || title.indexOf("投票一覧") >= 0 || bodyText.indexOf("場名から続けて入力") >= 0);

      if (isVoteListScreen) {
        if (st.betIndex + 1 < bets.length) {
          // 次の買い目がある場合 -> 「場名から続けて入力」を押して次へ
          dg("🏇 次の買い目（" + (st.betIndex + 2) + "/" + bets.length + "件目）へ進みます...");
          var fromJyoBtn = document.getElementById("fromjyo_top") ||
                           document.querySelector("a#fromjyo_top, #fromjyo_top a, a#fromjyo_under, #fromjyo_under a") ||
                           allLinks.find(function(a) {
                             var t = (a.innerText || a.textContent || "").trim();
                             return t.indexOf("場名から") >= 0 || t.indexOf("競馬場") >= 0;
                           });
          if (fromJyoBtn) {
            trigger(fromJyoBtn);
          } else if (typeof ToJyo === "function") {
            ToJyo();
          }
          st.betIndex++;
          st.actionCooldown = Date.now() + 1200;
          return { status: "NEXT_BET_TRIGGERED" };
        } else {
          // 全買い目セット完了時 -> 「入力終了」を押して最終確認画面(pw_741_i.cgi)へ
          if (!st._toSendTriggered) {
            dg("📑 【全" + bets.length + "件セット完了】 投票確認画面へ進みます...");
            st._toSendTriggered = true;
            var toSendBtn = document.getElementById("to_send_top") ||
                            document.querySelector("a#to_send_top, #to_send_top a, a#to_send_under, #to_send_under a") ||
                            allLinks.find(function(a) {
                              var t = (a.innerText || a.textContent || "").trim();
                              return t === "入力終了" || t.indexOf("入力終了") >= 0 || t.indexOf("投票確認") >= 0;
                            });
            if (typeof ToSend === "function") {
              ToSend();
            } else if (toSendBtn) {
              trigger(toSendBtn);
            }
            st.actionCooldown = Date.now() + 2000;
            return { status: "TO_SEND" };
          } else {
            dg("⏳ 【確認画面遷移中】 ページ読み込みを待機中...");
            return { status: "WAITING_CONFIRM" };
          }
        }
      }

      // 3. 金額入力画面 (#kin)
      var isAmountScreen = (curHash === "#kin" || (activePage && activePage.id === "kin") || title.indexOf("金額") >= 0 || !!document.getElementById("kin"));

      if (isAmountScreen) {
        var kinInput = activePage.querySelector("#kin input, input.amount, input[name='amount'], input[type='tel'], input[type='text']");
        var setBtn = allLinks.find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return t === "セット" || t.indexOf("セット") >= 0;
        }) || Array.from(document.querySelectorAll("a, button")).find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return t === "セット" || t.indexOf("セット") >= 0;
        });

        // マルチ指定の反映
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

      // 4. 馬番選択画面 (#uma1, #uma2, #umab 等)
      var isUma1 = (curHash === "#uma1" || title === "軸" || title.indexOf("軸") >= 0);
      var isUma2 = (curHash === "#uma2" || title === "相手" || title.indexOf("相手") >= 0);
      var isHorseScreen = isUma1 || isUma2 || curHash === "#umab" || (title.indexOf("馬番") >= 0 || title.indexOf("選択") >= 0) && !isAmountScreen && !isVoteListScreen;

      function matchHorseEl(el, hNum) {
        if (!el) return false;
        var t = (el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim();
        var hStr = String(parseInt(hNum, 10));
        var p1 = new RegExp("^0*" + hStr + "(?:\\s|$|[^0-9])");
        if (p1.test(t)) return true;
        var dv = el.getAttribute("data-value") || el.getAttribute("data-horse") || el.getAttribute("id") || "";
        if (dv === hStr || dv === ("0" + hStr) || dv === "horse_" + hStr) return true;
        return false;
      }

      if (isHorseScreen) {
        // A. 相手選択画面 (#uma2)
        if (isUma2) {
          dg("🐎 【相手馬選択】 相手馬を選択中...");
          var targetPartners = (partners && partners.length > 0) ? partners : (axes.length > 1 ? axes.slice(1) : []);
          var horseRows = Array.from(activePage.querySelectorAll("table tr, ul li, div.horse-row, a.ui-link"));

          targetPartners.forEach(function(h) {
            var el = allLinks.find(function(a) { return matchHorseEl(a, h); });
            if (!el) {
              el = Array.from(document.querySelectorAll("a, button, li.ui-btn")).find(function(a) { return matchHorseEl(a, h); });
            }
            if (el && !el.classList.contains("selected")) {
              trigger(el);
            }
          });

          // 「金額入力画面へ」ボタン
          var toKinBtn = document.getElementById("to_kin") ||
                         activePage.querySelector("a#to_kin, [id*='to_kin']") ||
                         allLinks.find(function(a) {
                           var t = (a.innerText || a.textContent || "").trim();
                           var h = a.getAttribute("href") || "";
                           return t.indexOf("金額入力") >= 0 || h.indexOf("#kin") >= 0;
                         }) || Array.from(document.querySelectorAll("a, button")).find(function(a) {
                           var t = (a.innerText || a.textContent || "").trim();
                           var h = a.getAttribute("href") || "";
                           return t.indexOf("金額入力") >= 0 || h.indexOf("#kin") >= 0;
                         });

          if (toKinBtn && !toKinBtn.classList.contains("disabled")) {
            trigger(toKinBtn);
            st.actionCooldown = Date.now() + 800;
            return { status: "TO_KIN" };
          }
          st.actionCooldown = Date.now() + 800;
          return { status: "PARTNERS_SELECTING" };
        }

        // B. 軸馬選択画面 (#uma1) または 通常・ボックス馬番画面
        if (isUma1 || isHorseScreen) {
          dg("🐎 【馬番選択】 軸馬・買い目を選択中...");
          var targetList = (axes && axes.length > 0) ? axes : partners;
          if (targetList.length === 0 && rawSteps.length > 3) targetList = rawSteps.slice(3);

          targetList.forEach(function(h) {
            var el = allLinks.find(function(a) { return matchHorseEl(a, h); });
            if (!el) {
              el = Array.from(document.querySelectorAll("a, button, li.ui-btn")).find(function(a) { return matchHorseEl(a, h); });
            }
            if (el) {
              trigger(el);
            }
          });

          // 通常・ボックスで「金額入力画面へ」がある場合
          var toKinBtn2 = document.getElementById("to_kin") ||
                          allLinks.find(function(a) {
                            var t = (a.innerText || a.textContent || "").trim();
                            var h = a.getAttribute("href") || "";
                            return t.indexOf("金額入力") >= 0 || h.indexOf("#kin") >= 0;
                          });
          if (toKinBtn2 && !toKinBtn2.classList.contains("disabled")) {
            trigger(toKinBtn2);
            st.actionCooldown = Date.now() + 800;
            return { status: "TO_KIN" };
          }

          st.actionCooldown = Date.now() + 800;
          return { status: "HORSE_CLICKED" };
        }
      }

      // 5. 方式選択画面 (#hou)
      var isHouScreen = (curHash === "#hou" || (activePage && activePage.id === "hou") || title.indexOf("方式") >= 0);

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
            if (numAxes >= 2) targetLabels = ["1･2着ながし", "1・2着ながし", "軸2頭ながし", "ながし"];
            else targetLabels = ["1着ながし", "軸1頭ながし", "ながし"];
          } else if (sikiCode === "7") { // 3連複
            if (numAxes >= 2) targetLabels = ["軸2頭ながし", "1･2着ながし", "ながし"];
            else targetLabels = ["軸1頭ながし", "1着ながし", "ながし"];
          } else if (sikiCode === "6") { // 馬単
            targetLabels = ["1着ながし", "ながし"];
          } else { // 馬連、ワイド、枠連
            targetLabels = ["軸1頭ながし", "ながし"];
          }
        } else {
          targetLabels = ["通常"];
        }

        dg("📐 【方式選択】 「" + targetLabels[0] + "」を選択中...");

        function matchHouBtn(el) {
          if (!el) return false;
          var t = (el.innerText || el.textContent || "").trim();
          return targetLabels.some(function(lbl) {
            return t === lbl || t.indexOf(lbl) >= 0;
          });
        }

        var matchedHou = allLinks.find(matchHouBtn);
        if (!matchedHou) {
          matchedHou = Array.from(document.querySelectorAll("a, button, li.ui-btn")).find(matchHouBtn);
        }

        if (matchedHou) {
          trigger(matchedHou);
          st.actionCooldown = Date.now() + 800;
          return { status: "HOU_CLICKED" };
        }
      }

      // 6. 式別選択画面 (#siki)
      var isSikiScreen = (curHash === "#siki" || (activePage && activePage.id === "siki") || title.indexOf("式別") >= 0);

      if (isSikiScreen) {
        var targets = sikiMap[String(sikiCode)] || ["単勝"];
        dg("🎯 【式別選択】 「" + targets[0] + "」を選択中...");
        var matchedSiki = allLinks.find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return targets.some(function(tgt) { return t === tgt || t.indexOf(tgt) >= 0; });
        }) || Array.from(document.querySelectorAll("a, button, li.ui-btn")).find(function(a) {
          var t = (a.innerText || a.textContent || "").trim();
          return targets.some(function(tgt) { return t === tgt || t.indexOf(tgt) >= 0; });
        });

        if (matchedSiki) {
          trigger(matchedSiki);
          st.actionCooldown = Date.now() + 800;
          return { status: "SIKI_CLICKED" };
        }
      }

      // 7. レース選択画面 (#race)
      var isRaceScreen = (curHash === "#race" || (activePage && activePage.id === "race") || title.indexOf("レース") >= 0);

      if (isRaceScreen) {
        var targetRStr = String(parseInt(raceNum, 10));
        dg("🏁 【レース選択】 " + targetRStr + "R を選択中...");

        function matchRaceBtn(el) {
          if (!el) return false;
          var t = (el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim();
          var r1 = new RegExp("^0*" + targetRStr + "\\s*R(?:\\s|$|[^0-9])", "i");
          if (r1.test(t)) return true;
          var r2 = new RegExp("^0*" + targetRStr + "\\s*レース(?:\\s|$)", "i");
          if (r2.test(t)) return true;
          if (t === targetRStr || t === ("0" + targetRStr)) return true;
          var dv = el.getAttribute("data-value") || el.getAttribute("data-race") || el.getAttribute("id") || "";
          if (dv === targetRStr || dv === ("0" + targetRStr) || dv === "race_" + targetRStr) return true;
          return false;
        }

        var matchedRace = allLinks.find(matchRaceBtn) || Array.from(document.querySelectorAll("a, button, li.ui-btn")).find(matchRaceBtn);
        if (matchedRace) {
          trigger(matchedRace);
          st.actionCooldown = Date.now() + 800;
          return { status: "RACE_CLICKED" };
        }
      }

      // 8. 競馬場名選択画面 (pw_740_i.cgi または #jyo)
      var isVenueScreen = (title.indexOf("競馬場") >= 0 || title.indexOf("場名") >= 0 || title === "競馬場名" || (curHref.indexOf("pw_740_i.cgi") >= 0 && !curHash));

      if (isVenueScreen) {
        dg("🏇 【競馬場選択】 「" + (venue || "開催場") + (weekday ? "(" + weekday + ")" : "") + "」を選択中...");

        // 曜日と競馬場名の両方マッチを最優先
        var matchedVenue = null;
        if (venue && weekday) {
          matchedVenue = allLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return t.indexOf(venue) >= 0 && t.indexOf(weekday) >= 0;
          });
        }
        if (!matchedVenue && venue) {
          matchedVenue = allLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return t.indexOf(venue) >= 0;
          });
        }
        if (!matchedVenue) {
          var docLinks = Array.from(document.querySelectorAll("a, button, li.ui-btn"));
          if (venue && weekday) {
            matchedVenue = docLinks.find(function(a) {
              var t = (a.innerText || a.textContent || "").trim();
              return t.indexOf(venue) >= 0 && t.indexOf(weekday) >= 0;
            });
          }
          if (!matchedVenue && venue) {
            matchedVenue = docLinks.find(function(a) {
              var t = (a.innerText || a.textContent || "").trim();
              return t.indexOf(venue) >= 0;
            });
          }
        }

        if (matchedVenue) {
          trigger(matchedVenue);
          st.actionCooldown = Date.now() + 800;
          return { status: "VENUE_CLICKED" };
        }
      }

      // 9. TOPメニュー画面（通常投票へ進む）
      var regBtn = document.querySelector(".ui-page-active a.ico_regular, a.ico_regular.ui-link, a.ico_regular, a[class*='ico_regular']") ||
                   allLinks.find(function(a) {
                     var t = (a.innerText || a.textContent || "").trim();
                     return t === "通常投票" || t.indexOf("通常投票") >= 0;
                   }) ||
                   Array.from(document.querySelectorAll("a")).find(function(a) {
                     var t = (a.innerText || a.textContent || "").trim();
                     return t === "通常投票" || t.indexOf("通常投票") >= 0;
                   });

      var isTopScreen = (curHref.indexOf("pw_732_i.cgi") >= 0 || !!regBtn || typeof ToQRBet === "function" || typeof ToSPBet === "function") && !isVenueScreen && !isRaceScreen && !isSikiScreen && !isHouScreen && !isHorseScreen && !isAmountScreen && !isConfirmScreen && !isVoteListScreen;

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

      // 10. 待機
      dg("⏳ 【画面読み込み中】 " + (title ? "タイトル: " + title : "処理待機中..."));
      return { status: "WAITING" };
    } catch(err) {
      dg("⚠️ エラー: " + err.message, "rgba(239,68,68,0.95)");
      return { status: "ERROR", error: err.message };
    }
  })();
  `;

  // 5. 実行監視ループ
  let isDone = false;
  let loopStart = Date.now();
  let maxTimeoutMs = Math.max(90000, bets.length * 45000);

  while (!isDone && (Date.now() - loopStart < maxTimeoutMs)) {
    try {
      let res = await wv.evaluateJavaScript(automationScript, false);
      if (res && res.status === "DONE") {
        isDone = true;
        break;
      }
    } catch(e) {}
    await new Promise(resolve => Timer.schedule(350, false, resolve));
  }

  // 6. WebViewの終了待機
  await presentPromise;
  Script.complete();
}

await main();
