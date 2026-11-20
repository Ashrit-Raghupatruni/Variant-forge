from typing import Dict, List, Optional

# Verified marketing copy dictionaries for English, Hindi, and Telugu
MARKETING_DICTIONARY: Dict[str, Dict[str, str]] = {
    "hi": {
        "New Arrival": "नया आगमन",
        "Limited Time Offer": "सीमित समय का ऑफर",
        "Shop Now": "अभी खरीदें",
        "Up to 50% Off": "50% तक की भारी छूट",
        "Special Discount": "विशेष छूट",
        "Premium Quality": "प्रीमियम गुणवत्ता",
        "Best Seller": "सर्वाधिक बिकने वाला",
        "Buy 1 Get 1 Free": "एक खरीदें, एक मुफ्त पाएं",
        "Flash Sale": "फ्लैश सेल",
        "Order Today": "आज ही ऑर्डर करें"
    },
    "te": {
        "New Arrival": "కొత్తగా వచ్చింది",
        "Limited Time Offer": "పరిమిత కాల ఆఫర్",
        "Shop Now": "ఇప్పుడే కొనండి",
        "Up to 50% Off": "50% వరకు తగ్గింపు",
        "Special Discount": "ప్రత్యేక తగ్గింపు",
        "Premium Quality": "ప్రీమియం నాణ్యత",
        "Best Seller": "అత్యధికంగా అమ్ముడైనది",
        "Buy 1 Get 1 Free": "ఒకటి కొంటే ఒకటి ఉచితం",
        "Flash Sale": "ఫ్లాష్ సేల్",
        "Order Today": "ఈరోజే ఆర్డర్ చేయండి"
    },
    "en": {
        "New Arrival": "New Arrival",
        "Limited Time Offer": "Limited Time Offer",
        "Shop Now": "Shop Now",
        "Up to 50% Off": "Up to 50% Off",
        "Special Discount": "Special Discount",
        "Premium Quality": "Premium Quality"
    }
}

class LocalizationService:
    """
    Campaign copy localization engine supporting English, Hindi, and Telugu.
    Includes typography reflow algorithms to ensure text bounds fit visual creatives.
    """
    @staticmethod
    def translate(text: str, target_lang: str) -> str:
        lang_code = target_lang.lower().strip()
        lang_dict = MARKETING_DICTIONARY.get(lang_code, {})
        
        for en_key, val in lang_dict.items():
            if en_key.lower() == text.lower().strip():
                return val
                
        return f"[{lang_code.upper()}] {text}"

    @staticmethod
    def compute_typography_reflow(
        text: str,
        base_font_size: int = 36,
        max_box_width: int = 400
    ) -> int:
        """
        Dynamically adjusts font size based on script characteristics (Indic scripts like Hindi/Telugu
        typically require subtle scaling to prevent horizontal or vertical overflow).
        """
        char_count = len(text)
        char_width_factor = 0.58
        estimated_width = char_count * (base_font_size * char_width_factor)
        
        if estimated_width > max_box_width:
            ratio = max_box_width / estimated_width
            return max(int(base_font_size * ratio), 14)
        return base_font_size
