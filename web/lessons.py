"""Short, original lessons. Examples are independent of research datasets."""

LESSONS = {
    "article_omission": {
        "rule": "介紹一個人的身分，或說明某個東西是什麼時，單數可數名詞前通常需要 a 或 an。看後面單字的第一個發音：子音開頭用 a，母音開頭用 an。這一課只練習 a/an，不處理需要上下文才能決定的 the。",
        "contrast": "中文的「她是攝影師」不需要冠詞；英文的單數可數名詞通常需要。這種結構差異可能讓學習者漏掉 a/an。",
        "examples": [
            {"before": "Nora is photographer.", "after": "Nora is a photographer.", "note": "photographer 是單數可數名詞，且以子音發音開頭，用 a。"},
            {"before": "This is empty jar.", "after": "This is an empty jar.", "note": "empty 以母音發音開頭，所以在 empty jar 前用 an。"},
        ],
    },
    "plural_marking_omission": {
        "rule": "數量大於一，或名詞前有 many、several、both 等明確的複數線索時，可數名詞也要改成複數。多數名詞加 -s，有些加 -es 或改變拼字。先找到同一個名詞片語裡的數量，再檢查名詞字尾。",
        "contrast": "中文說「四張卡片」時，卡片本身不變；英文的 four cards 則要改變名詞形式。這種差異可能讓人漏掉複數字尾。",
        "examples": [
            {"before": "I counted four cloud.", "after": "I counted four clouds.", "note": "four 大於一，cloud 要用複數 clouds。"},
            {"before": "We packed several box.", "after": "We packed several boxes.", "note": "several 表示好幾個，box 的複數是 boxes。"},
        ],
    },
    "past_tense_not_inflected": {
        "rule": "句子有 yesterday、last night、two days ago 等明確的過去時間，而且動作已經發生時，主要動詞要用過去式。規則動詞通常加 -ed；不規則動詞要記住對應形式，例如 go → went。這一課不練習完成式或過去分詞。",
        "contrast": "中文可以靠「昨天」交代時間，動詞不必改變；英文通常連動詞也要改成過去式。這可能是漏改動詞的原因之一。",
        "examples": [
            {"before": "Yesterday, I paint a kite.", "after": "Yesterday, I painted a kite.", "note": "Yesterday 是過去時間，paint 要改成 painted。"},
            {"before": "We go to the market last Sunday.", "after": "We went to the market last Sunday.", "note": "last Sunday 指過去，go 的過去式是 went。"},
        ],
    },
    "subject_verb_agreement": {
        "rule": "一般現在式中，he、she 或一個人的名字作主詞時，動作動詞通常加 -s/-es，例如 she reads。I、you、we、they 通常用原形。have/has 和 am/is/are 也要配合主詞。這一課不練習 do/does 或 was/were。",
        "contrast": "中文的「他跑步」和「他們跑步」使用相同的動詞；英文需要依主詞選擇 runs 或 run。這種差異可能讓動詞字尾被忽略。",
        "examples": [
            {"before": "She water the plants every day.", "after": "She waters the plants every day.", "note": "She 是第三人稱單數，一般現在式用 waters。"},
            {"before": "They has a small garden.", "after": "They have a small garden.", "note": "They 搭配 have；he/she 才搭配 has。"},
        ],
    },
    "be_verb_omission": {
        "rule": "英文描述主詞的狀態、身分或位置時，常需要 be 動詞來連接；描述正在進行的動作時，也需要 be + V-ing。現在式中 I 用 am，he/she/it 用 is，you/we/they 用 are。這一課練習補上漏掉的 be。",
        "contrast": "中文可以直接說「她很安靜」，不需要「是」；英文的 She is quiet. 需要 is。這種差異可能讓 be 動詞被省略。",
        "examples": [
            {"before": "She quiet today.", "after": "She is quiet today.", "note": "is 連接 She 和描述狀態的 quiet。"},
            {"before": "We waiting for the train now.", "after": "We are waiting for the train now.", "note": "正在等待用 are waiting，不能只有 waiting。"},
        ],
    },
    "to_omission": {
        "rule": "want、need、hope、plan 等動詞後接另一個動作時，通常使用 to + 原形動詞，例如 want to learn。另外，listen to、belong to，以及 go to + 地點等結構也需要 to。不要把這個規則套用到所有動詞或所有地點。",
        "contrast": "中文的「想學」和「聽音樂」可以直接連接；英文常需要 to。這種連接方式的差異可能讓 to 被漏掉。",
        "examples": [
            {"before": "I hope learn French.", "after": "I hope to learn French.", "note": "hope 後接動作時，用 to + 原形動詞。"},
            {"before": "We listen the radio every morning.", "after": "We listen to the radio every morning.", "note": "listen 接聆聽的對象時，需要介系詞 to。"},
        ],
    },
    "degree_adverb_misuse": {
        "rule": "very、super、extremely 不能直接放在 like、love、enjoy、want 這類動詞前。想表達「真的很喜歡」，可以用 really like；修飾形容詞時，very 則可以使用，例如 very happy。本課把這些動詞前不合適的程度副詞改成 really。",
        "contrast": "中文可以說「很喜歡」，但英文不能直接把「很」換成 very 放在 like 前。這種逐字對應可能造成錯誤，不代表每個人的錯誤都來自中文。",
        "examples": [
            {"before": "I very enjoy this class.", "after": "I really enjoy this class.", "note": "really 可以直接修飾 enjoy；very 不行。"},
            {"before": "She extremely likes this puzzle.", "after": "She really likes this puzzle.", "note": "extremely 不能直接修飾 likes，這裡可改成 really。"},
        ],
    },
}
