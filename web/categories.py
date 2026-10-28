"""Learner-facing metadata; IDs follow D-008 without importing the ML runtime."""

CATEGORIES = {
    "article_omission": {
        "name": "冠詞遺漏", "english": "Missing a/an", "symbol": "a / an",
        "group": "nouns", "group_name": "名詞與冠詞", "tone": "mint",
        "description": "一個人、一件事物，別忘了前面的 a 或 an。",
        "before": "Mia bought notebook.", "after": "Mia bought a notebook.",
        "hint": "notebook 是單數可數名詞，這裡需要 a。",
    },
    "plural_marking_omission": {
        "name": "複數標記", "english": "Plural marking", "symbol": "+ s",
        "group": "nouns", "group_name": "名詞與冠詞", "tone": "blue",
        "description": "不只一個的時候，讓名詞也跟著改變。",
        "before": "I packed three pencil.", "after": "I packed three pencils.",
        "hint": "three 表示三個，pencil 要改成複數 pencils。",
    },
    "past_tense_not_inflected": {
        "name": "過去式", "english": "Past tense", "symbol": "+ ed",
        "group": "verbs", "group_name": "動詞與時態", "tone": "peach",
        "description": "昨天發生的事，用過去式說清楚。",
        "before": "Yesterday, we watch a film.", "after": "Yesterday, we watched a film.",
        "hint": "Yesterday 指出過去的時間，watch 要改成 watched。",
    },
    "subject_verb_agreement": {
        "name": "主詞動詞一致", "english": "Subject-verb agreement", "symbol": "↔",
        "group": "verbs", "group_name": "動詞與時態", "tone": "lilac",
        "description": "留意誰做這件事，選對相應的動詞形式。",
        "before": "My sister collect stamps.", "after": "My sister collects stamps.",
        "hint": "My sister 是第三人稱單數，一般現在式用 collects。",
    },
    "be_verb_omission": {
        "name": "Be 動詞遺漏", "english": "Missing be-verb", "symbol": "be",
        "group": "verbs", "group_name": "動詞與時態", "tone": "blue",
        "description": "描述狀態或正在做的事，補上需要的 be 動詞。",
        "before": "The room quiet.", "after": "The room is quiet.",
        "hint": "這裡需要 is 連接主詞 The room 和形容詞 quiet。",
    },
    "to_omission": {
        "name": "To 的用法", "english": "Missing to", "symbol": "to →",
        "group": "patterns", "group_name": "句型與修飾", "tone": "mint",
        "description": "連接動作或對象時，找出不可少的 to。",
        "before": "We plan visit the museum.", "after": "We plan to visit the museum.",
        "hint": "plan 後接另一個動作時，使用 to + 原形動詞。",
    },
    "degree_adverb_misuse": {
        "name": "程度副詞誤用", "english": "Degree-adverb misuse", "symbol": "very?",
        "group": "patterns", "group_name": "句型與修飾", "tone": "warm",
        "description": "「很喜歡」怎麼說？分清 very 和 really 的位置。",
        "before": "I very enjoy this class.", "after": "I really enjoy this class.",
        "hint": "really 可以修飾 enjoy；very 不能直接放在 enjoy 前面。",
    },
}
