# Step 1: Read data

def read_raw_data():
    File = open("pr_data.txt", "r")
    s = File.read()
    File.close()

    s = list(s)
    i = 0

    while i < len(s):
        if ((ord(s[i]) >= ord('0') and ord(s[i]) <= ord('9')) or s[i] == "'" or s[i] == "\n" or s[i] == ' '):
            s.pop(i)
            continue

        if s[i] == '?' or s[i] == '_':
            s[i] = 'U'

        # FIX: convert Str → Seq
        if i + 2 < len(s) and s[i] == 'S' and s[i+1] == 't' and s[i+2] == 'r':
            s[i+1] = 'e'
            s[i+2] = 'q'
            i += 2

        i += 1

    s = "".join(s)
    s = s.split("Seq{}=")
    s.pop(0)

    sequence = []
    structure = []

    for i in range(len(s)):
        if i % 2 == 0:
            sequence.append(s[i])
        else:
            structure.append(s[i])

    return sequence, structure


# Step 2: Create sliding window dataset

def create_dataset(sequences, structures, window_size=5):
    X = []
    y = []

    half = window_size // 2

    for i in range(len(sequences)):
        seq = sequences[i]
        struc = structures[i]

        min_len = min(len(seq), len(struc))

        for j in range(half, min_len - half):
            window = seq[j-half : j+half+1]
            label = struc[j]

            X.append(window)
            y.append(label)

    return X, y


# Step 3: Encode using One-Hot

from sklearn.preprocessing import OneHotEncoder

def encode_windows(X):
    X_chars = [list(window) for window in X]

    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X_chars)

    return X_encoded


# Step 4: Run pipeline

print("Reading data...")
sequences, structures = read_raw_data()

print("Creating dataset...")
X, y = create_dataset(sequences, structures, window_size=5)

print("Encoding...")
X = encode_windows(X)

# Reduce dataset size for speed (temporary)
#X = X[:5000]
#y = y[:5000]

print("Training...")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))