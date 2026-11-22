// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: green; icon-glyph: horse;

/**
 * JRA スマッピー 自動入力スクリプト (for Scriptable)
 * 画面状態駆動型・複数買い目の一括連続入力・金額自動入力完全対応版
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

  // 入力JSONの正規化
  let bets = [];
  if (params) {
    if (params.bets && Array.isArray(params.bets)) {
      bets = params.bets;
    } else if (params.steps || params.venueName || params.placeName) {
      bets = [params];
    }
  }

  if (!bets || bets.length === 0) {
    let a = new Alert();
    a.title = "買い目データが見つかりません";
    a.message = "競馬予想サイト上の「スマッピー」ボタンから実行してください。";
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

  // 3. オートメーションスクリプト（画面状態駆動・外側から継続注入）
  let automationScript = `
  (function() {
    var bets = ${JSON.stringify(bets)};

    if (!window._smappyState) {
      window._smappyState = {
        betIndex: 0,
        lastStepName: "",
        actionCooldown: 0,
        amountSetAttempted: false
      };
    }
    var st = window._smappyState;

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
      var x = document.getElementById("smappy-diag");
      if (!x) {
        x = document.createElement("div");
        x.id = "smappy-diag";
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:2147483647;background:" + (col || "rgba(16,185,129,0.95)") + ";color:#ffffff;font-size:12px;font-weight:bold;padding:9px 12px;pointer-events:none;text-align:center;box-shadow:0 3px 10px rgba(0,0,0,0.35);line-height:1.4;";
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

      try {
        var r = el.getBoundingClientRect();
        var x = r.left + r.width / 2;
        var y = r.top + r.height / 2;
        var t = new Touch({identifier: Date.now(), target: el, clientX: x, clientY: y, radiusX: 2, radiusY: 2});
        var to = {bubbles: true, cancelable: true, touches: [t], targetTouches: [t], changedTouches: [t], view: window};
        el.dispatchEvent(new TouchEvent("touchstart", to));
        el.dispatchEvent(new TouchEvent("touchend", to));
      } catch (err) {}

      try {
        var r2 = el.getBoundingClientRect();
        var x2 = r2.left + r2.width / 2;
        var y2 = r2.top + r2.height / 2;
        el.dispatchEvent(new PointerEvent("pointerdown", {bubbles: true, cancelable: true, clientX: x2, clientY: y2, view: window}));
        el.dispatchEvent(new PointerEvent("pointerup", {bubbles: true, cancelable: true, clientX: x2, clientY: y2, view: window}));
      } catch (err) {}

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

      try {
        if (window.$ && typeof window.$(el).trigger === "function") {
          window.$(el).trigger("vclick");
          window.$(el).trigger("tap");
          window.$(el).trigger("click");
        }
      } catch (err) {}

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

      // 全リンク・ボタンの取得
      var allLinks = Array.from(activePage.querySelectorAll("a, button, input[type='button'], input[type='submit'], li.ui-btn, [role='button']"));
      var titleEl = activePage.querySelector(".ui-title, h1.ui-title") || document.querySelector(".ui-page-active .ui-title, .ui-title");
      var title = titleEl ? (titleEl.innerText || titleEl.textContent || "").trim() : "";

      // 1. 投票一覧画面 (#toui)
      var isVoteListScreen = (curHash === "#toui" || (activePage && activePage.id === "toui") || title.indexOf("投票一覧") >= 0 || bodyText.indexOf("場名から続けて入力") >= 0);

      if (isVoteListScreen) {
        st.amountSetAttempted = false; // 次の買い目のためにフラグリセット

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
          // 全買い目セット完了時 -> 「QR作成」または「投票用QR表示」
          dg("🎉 【全" + bets.length + "件セット完了】 QR作成ボタンを押してQRコードを表示してください！", "rgba(16,185,129,0.95)");
          var qrBtn = allLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return t.indexOf("QR") >= 0 || t.indexOf("ＱＲ") >= 0 || t.indexOf("作成") >= 0;
          });
          if (qrBtn) {
            trigger(qrBtn);
          }
          return { status: "DONE" };
        }
      }

      // 2. 金額入力画面 (#kin または タイトル「金額入力」)
      var kinInputs = Array.from(activePage.querySelectorAll("#kin input, input.amount, input[name='amount'], input[type='tel'], input[type='text'], input[type='number']"));
      var isAmountScreen = (curHash === "#kin" || (activePage && activePage.id === "kin") || title.indexOf("金額") >= 0 || (kinInputs.length > 0 && bodyText.indexOf("00円") >= 0));

      if (isAmountScreen) {
        var kinInput = kinInputs.find(function(inp) {
          var r = inp.getBoundingClientRect();
          return r.width > 0 && r.height > 0 && !inp.disabled;
        }) || kinInputs[0];

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
          kinInput.dispatchEvent(new Event('input', {bubbles:true}));
          kinInput.dispatchEvent(new Event('change', {bubbles:true}));
        }
        if (typeof VMA === "function") {
          try { VMA(String(hundreds)); } catch(e) {}
        }

        setTimeout(function() {
          if (typeof SetBet === "function") {
            try { SetBet(0); } catch(e) {}
          } else if (setBtn) {
            trigger(setBtn);
          }
        }, 150);

        st.actionCooldown = Date.now() + 1000;
        return { status: "AMOUNT_SET" };
      }

      // 3. 馬番選択画面 (#uma1, #uma2, #umab 等)
      var isUma1 = (curHash === "#uma1" || title === "軸" || title.indexOf("軸") >= 0 || title.indexOf("1着") >= 0);
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
        // A. 相手馬選択画面 (#uma2)
        if (isUma2) {
          dg("🐎 【相手馬選択】 相手馬を選択中...");
          var targetPartners = (partners && partners.length > 0) ? partners : (axes.length > 1 ? axes.slice(1) : []);
          targetPartners.forEach(function(h) {
            var el = allLinks.find(function(a) { return matchHorseEl(a, h); }) ||
                     Array.from(document.querySelectorAll("a, button, li.ui-btn")).find(function(a) { return matchHorseEl(a, h); });
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
                           return t.indexOf("金額入力") >= 0 || h.indexOf("#kin") >= 0 || t.indexOf("次へ") >= 0;
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
            var el = allLinks.find(function(a) { return matchHorseEl(a, h); }) ||
                     Array.from(document.querySelectorAll("a, button, li.ui-btn")).find(function(a) { return matchHorseEl(a, h); });
            if (el) {
              trigger(el);
            }
          });

          // 金額入力へ進むボタンがある場合（通常・ボックスなど）
          var toKinBtn2 = document.getElementById("to_kin") ||
                          allLinks.find(function(a) {
                            var t = (a.innerText || a.textContent || "").trim();
                            var h = a.getAttribute("href") || "";
                            return t.indexOf("金額入力") >= 0 || h.indexOf("#kin") >= 0 || t.indexOf("次へ") >= 0;
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

      // 4. 方式選択画面 (#hou)
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

        var matchedHou = allLinks.find(matchHouBtn) || Array.from(document.querySelectorAll("a, button, li.ui-btn")).find(matchHouBtn);
        if (matchedHou) {
          trigger(matchedHou);
          st.actionCooldown = Date.now() + 800;
          return { status: "HOU_CLICKED" };
        }
      }

      // 5. 式別選択画面 (#siki)
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

      // 6. レース選択画面 (#race)
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

      // 7. 競馬場名選択画面 (#jyo または pw_982_i.cgi)
      var isVenueScreen = (title.indexOf("競馬場") >= 0 || title.indexOf("場名") >= 0 || title === "競馬場名" || (curHref.indexOf("pw_982_i.cgi") >= 0 && !curHash));

      if (isVenueScreen) {
        dg("🏇 【競馬場選択】 「" + (venue || "開催場") + (weekday ? "(" + weekday + ")" : "") + "」を選択中...");

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

      // 8. TOPメニュー画面（通常投票へ進む）
      var regBtn = document.querySelector(".ui-page-active a.ico_regular, a.ico_regular.ui-link, a.ico_regular, a[class*='ico_regular']") ||
                   allLinks.find(function(a) {
                     var t = (a.innerText || a.textContent || "").trim();
                     return t === "通常投票" || t.indexOf("通常投票") >= 0;
                   }) ||
                   Array.from(document.querySelectorAll("a")).find(function(a) {
                     var t = (a.innerText || a.textContent || "").trim();
                     return t === "通常投票" || t.indexOf("通常投票") >= 0;
                   });

      var isTopScreen = (curHref.indexOf("pw_982_i.cgi") < 0 && (!!regBtn || typeof ToQRBet === "function")) && !isVenueScreen && !isRaceScreen && !isSikiScreen && !isHouScreen && !isHorseScreen && !isAmountScreen && !isVoteListScreen;

      if (isTopScreen) {
        dg("📋 【TOP画面】 通常投票へ進みます...");
        if (typeof ToQRBet === "function") {
          try { ToQRBet(); } catch(e) {}
        } else if (regBtn) {
          trigger(regBtn);
        }
        st.actionCooldown = Date.now() + 1200;
        return { status: "TOP_CLICKED" };
      }

      dg("⏳ 【画面読み込み中】 " + (title ? "タイトル: " + title : "処理待機中..."));
      return { status: "WAITING" };
    } catch(err) {
      dg("⚠️ エラー: " + err.message, "rgba(239,68,68,0.95)");
      return { status: "ERROR", error: err.message };
    }
  })();
  `;

  // 4. 外側（Scriptable側）から定期注入・実行監視ループ
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

  await presentPromise;
  Script.complete();
}

await main();
