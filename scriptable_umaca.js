// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: purple; icon-glyph: id-card;

/**
 * JRA UMACA スマート自動投票スクリプト (for Scriptable)
 * 予想サイトの「UMACA」ボタンから起動します。
 * ログイン -> 通常投票 -> 競馬場 -> レース -> 式別 -> 方式 -> 馬番 -> 金額セット -> 投票確認画面まで完全自動実行します。
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

  if (!params) {
    let a = new Alert();
    a.title = "買い目データが見つかりません";
    a.message = "予想サイト上の「UMACA」ボタンから実行してください。";
    a.addAction("OK");
    await a.present();
    Script.complete();
    return;
  }

  const { steps, venueName, weekday, unitAmount, totalAmount, round, siki, hou, axes, partners, isMulti } = params;
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

  // 4. ログイン実行スクリプト
  let loginScript = `
  (function() {
    var c = ${JSON.stringify(cardNo)};
    var b = ${JSON.stringify(birthDay)};
    var p = ${JSON.stringify(passNo)};

    function setDiag(m, col) {
      var x = document.getElementById("umaca-diag");
      if (!x) {
        x = document.createElement("div");
        x.id = "umaca-diag";
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:100000;background:" + (col || "rgba(147,51,234,0.95)") + ";color:#ffffff;font-size:12px;font-weight:bold;padding:8px 12px;pointer-events:none;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.3);";
        document.body.appendChild(x);
      }
      x.innerText = m;
    }
    setDiag("🟣 UMACAへログイン中...");

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
      elC.value = c;
      elB.value = b;
      elP.value = p;
      if (elC.onchange) elC.onchange();
      if (elB.onchange) elB.onchange();
      if (elP.onchange) elP.onchange();

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

  // 5. メニュー画面または通常投票画面が表示されるまで待機（最大12秒ポーリング）
  let menuReached = false;
  for (let waitCount = 0; waitCount < 30; waitCount++) {
    await new Promise(res => Timer.schedule(400, false, res));
    let checkMenuScript = `
    (function() {
      if (document.getElementById("jyo") || document.getElementById("race") || document.getElementById("siki") || !!document.querySelector("ul.selectList a, #jyo a")) {
        return "VOTING_SCREEN";
      }
      if (typeof ToSPBet === "function" || !!document.querySelector("a.ico_regular") || !!Array.from(document.querySelectorAll("a,button")).find(b => (b.textContent||"").includes("通常投票"))) {
        return "MENU_SCREEN";
      }
      if (document.body && (document.body.innerText.indexOf("エラー") >= 0 || document.body.innerText.indexOf("誤りがあります") >= 0)) {
        return "LOGIN_ERROR";
      }
      return "WAITING";
    })();
    `;
    let status = await wv.evaluateJavaScript(checkMenuScript, false);
    if (status === "VOTING_SCREEN" || status === "MENU_SCREEN") {
      menuReached = (status === "MENU_SCREEN");
      break;
    }
    if (status === "LOGIN_ERROR") {
      let errA = new Alert();
      errA.title = "UMACAログインエラー";
      errA.message = "カード番号、生年月日、または暗証番号に誤りがあります。\nKeychainの情報を再設定してください。";
      errA.addAction("設定を再入力する");
      errA.addCancelAction("キャンセル");
      let choice = await errA.present();
      if (choice === 0) {
        Keychain.remove("umaca_card_no");
        Keychain.remove("umaca_birth_day");
        Keychain.remove("umaca_pass_no");
      }
      Script.complete();
      return;
    }
  }

  // 6. メニュー画面から「通常投票」へ遷移
  if (menuReached) {
    let toVoteScript = `
    (function() {
      function setDiag(m) {
        var x = document.getElementById("umaca-diag");
        if (!x) {
          x = document.createElement("div");
          x.id = "umaca-diag";
          x.style = "position:fixed;top:0;left:0;width:100%;z-index:100000;background:rgba(147,51,234,0.95);color:#ffffff;font-size:12px;font-weight:bold;padding:8px 12px;pointer-events:none;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.3);";
          document.body.appendChild(x);
        }
        x.innerText = m;
      }
      setDiag("📋 メニュー検出！通常投票へ遷移中...");

      if (typeof ToSPBet === "function") {
        ToSPBet(0);
      } else {
        var a = document.querySelector("a.ico_regular");
        if (a) {
          var href = a.getAttribute("href") || "";
          if (href.toLowerCase().startsWith("javascript:")) {
            try { eval(href.replace(/^javascript:/i, '')); return; } catch(e) {}
          }
          a.click();
        } else {
          var regularBtn = Array.from(document.querySelectorAll("a,button")).find(b => (b.textContent||"").includes("通常投票"));
          if (regularBtn) regularBtn.click();
        }
      }
    })();
    `;
    await wv.evaluateJavaScript(toVoteScript, false);

    // 通常投票画面が表示されるまで待機（最大10秒ポーリング）
    for (let waitCount = 0; waitCount < 25; waitCount++) {
      await new Promise(res => Timer.schedule(400, false, res));
      let inVoting = await wv.evaluateJavaScript(`
        (function() {
          return !!(document.getElementById("jyo") || document.getElementById("race") || document.getElementById("siki") || document.querySelector("ul.selectList a, #jyo a"));
        })();
      `, false);
      if (inVoting) break;
    }
  }

  await new Promise(res => Timer.schedule(600, false, res));

  // 7. 通常投票画面での高精度シーケンシャル自動選択スクリプト
  let runnerScript = `
  (function() {
    var params = ${JSON.stringify(params)};
    var placeName = ${JSON.stringify(venueName || params.placeName || "")};
    var raceNo = ${JSON.stringify(round || params.raceNo || (steps && steps[1]) || "")};
    var sikiCode = ${JSON.stringify(siki || (steps && steps[2]) || "1")};
    var houCode = ${JSON.stringify(hou || (steps && steps[3]) || "0")};
    var rawAxes = ${JSON.stringify(axes || [])};
    var rawPartners = ${JSON.stringify(partners || [])};
    var rawSteps = ${JSON.stringify(steps || [])};
    var hundreds = ${hundreds};
    var totalAmount = ${tAmount};
    var passNo = ${JSON.stringify(passNo)};
    var isMulti = ${Boolean(isMulti)};

    // 式別マップ
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
        x.style = "position:fixed;top:0;left:0;width:100%;z-index:100000;background:" + (col || "rgba(147,51,234,0.95)") + ";color:#ffffff;font-size:12px;font-weight:bold;padding:8px 12px;pointer-events:none;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.3);";
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

    // ステートマシンによる順次実行
    var step = 1;
    var retryCount = 0;
    var startTime = Date.now();

    function stepLoop() {
      if (Date.now() - startTime > 35000) {
        dg("⚠️ 処理がタイムアウトしました", "rgba(239,68,68,0.95)");
        return;
      }

      try {
        // Step 1: 競馬場選択
        if (step === 1) {
          dg("🏇 [Step 1/8] 競馬場を選択中: " + (placeName || "会場"));
          var jyoLinks = Array.from(document.querySelectorAll("ul.selectList a, #jyo a, a"));
          var matchedJyo = null;

          if (placeName) {
            matchedJyo = jyoLinks.find(function(a) {
              return (a.innerText || a.textContent || "").indexOf(placeName) >= 0;
            });
          }
          if (!matchedJyo && rawSteps.length > 0) {
            var idx = parseInt(rawSteps[0]);
            if (!isNaN(idx) && jyoLinks[idx]) matchedJyo = jyoLinks[idx];
          }

          if (matchedJyo) {
            trigger(matchedJyo);
            step = 2;
            retryCount = 0;
            setTimeout(stepLoop, 600);
            return;
          }

          // すでにレース画面または式別画面にいる場合
          if (document.getElementById("race") || document.querySelector("ul.selectList a, #race a")) {
            step = 2;
            retryCount = 0;
            setTimeout(stepLoop, 200);
            return;
          }
        }

        // Step 2: レース選択
        else if (step === 2) {
          var rTarget = String(raceNo).replace(/[^0-9]/g, "");
          dg("🏁 [Step 2/8] レースを選択中: " + rTarget + "R");
          var raceLinks = Array.from(document.querySelectorAll("ul.selectList a, #race a, a"));
          var matchedRace = raceLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return t === rTarget + "R" || t === rTarget + "レース" || t.indexOf(rTarget + "R") >= 0;
          });

          if (matchedRace) {
            trigger(matchedRace);
            step = 3;
            retryCount = 0;
            setTimeout(stepLoop, 600);
            return;
          }

          // すでに式別画面にいる場合
          if (document.getElementById("siki") || document.querySelector("#siki a")) {
            step = 3;
            retryCount = 0;
            setTimeout(stepLoop, 200);
            return;
          }
        }

        // Step 3: 式別選択
        else if (step === 3) {
          var targets = sikiMap[String(sikiCode)] || ["単勝"];
          dg("🎯 [Step 3/8] 式別を選択中: " + targets[0]);
          var sikiLinks = Array.from(document.querySelectorAll("ul.selectList a, #siki a, a"));
          var matchedSiki = sikiLinks.find(function(a) {
            var t = (a.innerText || a.textContent || "").trim();
            return targets.some(function(target) { return t.indexOf(target) >= 0; });
          });

          if (matchedSiki) {
            trigger(matchedSiki);
            step = 4;
            retryCount = 0;
            setTimeout(stepLoop, 600);
            return;
          }

          // 方式または馬番画面にいる場合
          if (document.getElementById("hou") || document.querySelector("#hou a") || document.querySelector("input[name='multi']")) {
            step = 4;
            retryCount = 0;
            setTimeout(stepLoop, 200);
            return;
          }
        }

        // Step 4: 方式選択 (通常/ボックス/ながし)
        else if (step === 4) {
          var isSimple = (sikiCode === "1" || sikiCode === "2" || sikiCode === "9");
          var houLinks = Array.from(document.querySelectorAll("ul.selectList a, #hou a, a"));
          var hasHouScreen = document.getElementById("hou") || houLinks.some(function(a) {
            var t = (a.innerText || a.textContent || "");
            return t.indexOf("通常") >= 0 || t.indexOf("ボックス") >= 0 || t.indexOf("ながし") >= 0;
          });

          if (!isSimple && hasHouScreen) {
            var hName = "通常";
            if (houCode === "1" || houCode === "box") hName = "ボックス";
            else if (houCode === "2" || houCode === "nagashi" || houCode === "multi") hName = "ながし";

            dg("📐 [Step 4/8] 方式を選択中: " + hName);
            var matchedHou = houLinks.find(function(a) {
              return (a.innerText || a.textContent || "").indexOf(hName) >= 0;
            });

            if (matchedHou) {
              trigger(matchedHou);
              step = 5;
              retryCount = 0;
              setTimeout(stepLoop, 600);
              return;
            }
          } else {
            step = 5;
            retryCount = 0;
            setTimeout(stepLoop, 200);
            return;
          }
        }

        // Step 5: 馬番選択
        else if (step === 5) {
          dg("🐎 [Step 5/8] 馬番を選択中...");
          var allHorses = [];
          if (rawAxes && rawAxes.length > 0) allHorses = allHorses.concat(rawAxes);
          if (rawPartners && rawPartners.length > 0) allHorses = allHorses.concat(rawPartners);
          if (allHorses.length === 0 && rawSteps.length > 3) {
            allHorses = rawSteps.slice(3);
          }

          var hLinks = Array.from(document.querySelectorAll("a, button"));
          var clickedAny = false;

          var targetList = (rawAxes && rawAxes.length > 0) ? rawAxes : allHorses;
          targetList.forEach(function(h) {
            var hStr = String(parseInt(h));
            var el = hLinks.find(function(a) {
              var t = (a.innerText || a.textContent || "").trim();
              var dv = a.getAttribute("data-value");
              return t === hStr || dv === hStr || t === ("0" + hStr);
            });
            if (el) {
              trigger(el);
              clickedAny = true;
            } else if (typeof SelectHorse === "function") {
              SelectHorse(hStr);
              clickedAny = true;
            }
          });

          // ながし方式の場合、相手馬選択へ進む
          if (rawAxes && rawAxes.length > 0 && rawPartners && rawPartners.length > 0) {
            var nextBtn = Array.from(document.querySelectorAll("a, button")).find(function(a) {
              var t = (a.innerText || a.textContent || "");
              return t.indexOf("相手") >= 0 || t.indexOf("次へ") >= 0;
            });
            if (nextBtn) {
              trigger(nextBtn);
              setTimeout(function() {
                var pLinks = Array.from(document.querySelectorAll("a, button"));
                rawPartners.forEach(function(h) {
                  var hStr = String(parseInt(h));
                  var el = pLinks.find(function(a) {
                    var t = (a.innerText || a.textContent || "").trim();
                    var dv = a.getAttribute("data-value");
                    return t === hStr || dv === hStr;
                  });
                  if (el) trigger(el);
                  else if (typeof SelectHorse === "function") SelectHorse(hStr);
                });

                if (isMulti) {
                  var multiBox = document.querySelector("input[name='multi'], #multi, input[type='checkbox']");
                  if (multiBox && !multiBox.checked) {
                    multiBox.click();
                  }
                }
              }, 400);
            }
          }

          step = 6;
          retryCount = 0;
          setTimeout(stepLoop, 700);
          return;
        }

        // Step 6: 金額入力 & セット
        else if (step === 6) {
          dg("💰 [Step 6/8] 金額を入力中 (1点: " + (hundreds * 100) + "円)...");
          var kinInput = document.querySelector("#kin .amount input, input.amount, input[name='amount'], input[type='tel']");
          if (kinInput) {
            kinInput.value = String(hundreds);
            if (kinInput.onchange) kinInput.onchange();
            if (kinInput.oninput) kinInput.oninput();
          }
          if (typeof VMA === "function") {
            VMA(String(hundreds));
          }

          setTimeout(function() {
            dg("セットボタンを実行中...");
            if (typeof SetBet === "function") {
              SetBet(0);
            } else {
              var setBtn = Array.from(document.querySelectorAll("a, button")).find(function(b) {
                var t = (b.innerText || b.textContent || "");
                return t.indexOf("セット") >= 0 || t.indexOf("決定") >= 0;
              });
              if (setBtn) trigger(setBtn);
            }

            step = 7;
            retryCount = 0;
            setTimeout(stepLoop, 800);
          }, 400);
          return;
        }

        // Step 7: 投票確認画面へ遷移
        else if (step === 7) {
          dg("📑 [Step 7/8] 投票確認画面へ移動中...");
          if (typeof ToSend === "function") {
            ToSend();
          } else {
            var sendBtn = Array.from(document.querySelectorAll("a, button")).find(function(b) {
              var t = (b.innerText || b.textContent || "");
              return t.indexOf("入力終了") >= 0 || t.indexOf("投票確認") >= 0 || t.indexOf("終了") >= 0;
            });
            if (sendBtn) trigger(sendBtn);
          }

          step = 8;
          retryCount = 0;
          setTimeout(stepLoop, 1000);
          return;
        }

        // Step 8: 投票確認画面での入力 (合計金額 & 暗証番号)
        else if (step === 8) {
          var passInput = document.querySelector("input[type='password'], #password, input[name='p']");
          var inputs = Array.from(document.querySelectorAll("input"));
          var hasInputs = passInput || inputs.length > 0;

          if (hasInputs) {
            for (var j = 0; j < inputs.length; j++) {
              var inp = inputs[j];
              var itype = (inp.type || "").toLowerCase();
              var iid = (inp.id || "").toLowerCase();
              var iname = (inp.name || "").toLowerCase();
              if ((itype === "tel" || itype === "text" || itype === "number") && iid.indexOf("pass") < 0 && iname.indexOf("pass") < 0) {
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
        }

        retryCount++;
        setTimeout(stepLoop, 300);
      } catch (err) {
        dg("エラー: " + err.message, "rgba(239,68,68,0.95)");
      }
    }

    stepLoop();
  })();
  `;

  await wv.evaluateJavaScript(runnerScript, false);

  // 8. WebViewの終了待機
  await presentPromise;
  Script.complete();
}

await main();
