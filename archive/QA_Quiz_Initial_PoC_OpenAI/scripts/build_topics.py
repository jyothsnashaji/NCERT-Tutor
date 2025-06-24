import os,json

from pathlib import Path
TOPIC_FILE = Path("data/topics.json")
data_dir = Path("data/")

topics = {}

def save_topics():
    with open(TOPIC_FILE, "w") as f:
        # Convert tuple keys to JSON strings
        json.dump({json.dumps(k): v for k, v in topics.items()}, f, indent=2)

def add_topic(grade, subject, new_topic):
    key = (grade, subject)
    topics.setdefault(key, [])
    if new_topic not in topics[key]:
        topics[key].append(new_topic)

def get_topics(grade, subject):
    return topics.get((grade, subject), [])

def populate_topics():
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)

                # Extract grade, subject, chapter from path
                parts = file_path.split(os.sep)
                grade = parts[-3].replace("grade_", "")  # e.g., '11'
                subject = parts[-2].lower()              # e.g., 'physics'
                chapter = os.path.splitext(file)[0].lower()  # e.g., 'chapter1'
                print(f"Grade: {grade}, Subject: {subject}, Chapter: {chapter}")
                add_topic(grade, subject, chapter)

    # print the topics dictionary
    print(f"[INFO] Topics extracted: {topics}")
    save_topics()

if __name__ == "__main__":
    print("[INFO] Populating topics...")
    populate_topics()