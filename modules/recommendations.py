def get_recommendations(score):
    """
    Returns a dictionary of action plan recommendations based on the PHQ-9 score (0-27).
    Tiers:
    - Minimal (0-4)
    - Mild/Moderate (5-14)
    - Severe (15-27)
    """
    # Ensure score is an integer
    try:
        score = int(score)
    except:
        score = 0

    if score <= 4:
        return {
            "tier": "Minimal",
            "video_url": "https://www.youtube.com/embed/WPPPFqsECz0", # TED-Ed: How stress affects your brain
            "video_title": "Maintaining Mental Hygiene",
            "meditation": {
                "title": "5-Minute Daily Reboot",
                "desc": "A simple habit to keep your mind sharp and stress-free.",
                "steps": [
                    "Find a quiet space and sit comfortably.",
                    "Take 5 deep breaths (inhale for 4s, hold for 4s, exhale for 6s).",
                    "Mentally list 3 positive things that happened today."
                ]
            },
            "tricks": [
                "Drink at least 2 litres of water.",
                "Spend 15 minutes outside in the morning sunlight.",
                "Write down tomorrow's to-do list before bed to clear your mind.",
                "Limit screen time 1 hour before sleeping."
            ]
        }
    elif score <= 14:
        return {
            "tier": "Moderate",
            "video_url": "https://www.youtube.com/embed/1I9ADpXG6cE", # How to cope with anxiety/depression (general)
            "video_title": "Coping with Low Moods",
            "meditation": {
                "title": "10-Minute Guided Mindfulness",
                "desc": "Designed to gently pull you out of an anxious or low state.",
                "steps": [
                    "Lie down or sit with your back supported.",
                    "Focus entirely on the physical sensation of your breath entering your nose.",
                    "When your mind wanders (and it will), gently bring focus back to the breath without judging yourself.",
                    "Do a 'body scan' from your toes to your head, relaxing each muscle group."
                ]
            },
            "tricks": [
                "Go for a 20-minute walk without your phone.",
                "Reach out to one friend or family member today, even just a text.",
                "Break large tasks into tiny, 5-minute 'micro-tasks'.",
                "Listen to your favourite uplifting music or a calming podcast."
            ]
        }
    else:
        # 15-27: Severe
        return {
            "tier": "Severe",
            "video_url": "https://www.youtube.com/embed/mMRrCYPxD0I", # Grounding technique video or similar
            "video_title": "Managing Overwhelming Distress",
            "meditation": {
                "title": "The 5-4-3-2-1 Grounding Technique",
                "desc": "An immediate tool to pull your brain out of spiralling thoughts and into the present.",
                "steps": [
                    "Acknowledge 5 things you can SEE around you.",
                    "Acknowledge 4 things you can physically FEEL.",
                    "Acknowledge 3 things you can HEAR.",
                    "Acknowledge 2 things you can SMELL.",
                    "Acknowledge 1 thing you can TASTE."
                ]
            },
            "tricks": [
                "Be extremely kind to yourself. You are dealing with a heavy burden.",
                "Focus ONLY on the next 10 minutes. Don't worry about tomorrow.",
                "Do one tiny self-care act: drink a glass of water, or wash your face.",
                "Please consider reaching out to a professional therapist or a local mental health helpline."
            ]
        }
