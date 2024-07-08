#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>

// TF-IDF vector representation for profiles
struct TFIDFVector {
    std::vector<double> tfidf_values;
};

// TF-IDF vectorization
class TFIDFVectorizer {
private:
    std::vector<std::string> vocabulary;
    std::unordered_map<std::string, int> word_to_index;
public:
    TFIDFVectorizer(const std::vector<std::string>& profiles) {
        // Construct vocabulary
        for (const auto& profile : profiles) {
            for (const auto& word : tokenize(profile)) {
                if (word_to_index.find(word) == word_to_index.end()) {
                    word_to_index[word] = vocabulary.size();
                    vocabulary.push_back(word);
                }
            }
        }
    }

    TFIDFVector transform(const std::string& profile) const {
        TFIDFVector vector;
        vector.tfidf_values.resize(vocabulary.size(), 0.0);
        std::unordered_map<std::string, int> term_frequency;

        // Calculate term frequency
        for (const auto& word : tokenize(profile)) {
            term_frequency[word]++;
        }

        // Calculate TF-IDF values
        for (const auto& pair : term_frequency) {
            double tf = static_cast<double>(pair.second) / term_frequency.size();
            double idf = log(static_cast<double>(vocabulary.size()) / (word_to_index.at(pair.first) + 1));
            vector.tfidf_values[word_to_index.at(pair.first)] = tf * idf;
        }

        return vector;
    }

    // Tokenize profile into words
    std::vector<std::string> tokenize(const std::string& profile) const {
        std::vector<std::string> tokens;
        std::string token;
        for (char c : profile) {
            if (c == ' ' || c == '.' || c == ',') {
                if (!token.empty()) {
                    tokens.push_back(token);
                    token.clear();
                }
            } else {
                token.push_back(c);
            }
        }
        if (!token.empty()) {
            tokens.push_back(token);
        }
        return tokens;
    }
};

// Cosine similarity calculation
double cosine_similarity(const TFIDFVector& v1, const TFIDFVector& v2) {
    double dot_product = 0.0;
    double norm_v1 = 0.0;
    double norm_v2 = 0.0;

    for (size_t i = 0; i < v1.tfidf_values.size(); ++i) {
        dot_product += v1.tfidf_values[i] * v2.tfidf_values[i];
        norm_v1 += pow(v1.tfidf_values[i], 2);
        norm_v2 += pow(v2.tfidf_values[i], 2);
    }

    if (norm_v1 == 0 || norm_v2 == 0) {
        return 0.0; // Handle division by zero
    }

    return dot_product / (sqrt(norm_v1) * sqrt(norm_v2));
}

int main() {
    // Step 1: User and employer profiles
    std::vector<std::string> user_profiles = {
        "Experienced software developer with expertise in Python and machine learning, seeking remote positions.",
        "Recent graduate with a degree in finance and strong analytical skills, looking for entry-level positions in banking."
    };

    std::vector<std::string> employer_profiles = {
        "Tech startup seeking skilled developers with experience in web development and cloud computing.",
        "Financial institution looking for motivated graduates with a background in finance and a willingness to learn."
    };

    // Step 2: TF-IDF vectorization
    TFIDFVectorizer vectorizer(user_profiles);
    std::vector<TFIDFVector> user_vectors;
    for (const auto& profile : user_profiles) {
        user_vectors.push_back(vectorizer.transform(profile));
    }

    std::vector<TFIDFVector> employer_vectors;
    for (const auto& profile : employer_profiles) {
        employer_vectors.push_back(vectorizer.transform(profile));
    }

    // Step 3: Matching user profiles with job opportunities
    for (size_t i = 0; i < user_vectors.size(); ++i) {
        std::cout << "Recommended jobs for User " << i + 1 << ":\n";
        for (size_t j = 0; j < employer_vectors.size(); ++j) {
            double similarity = cosine_similarity(user_vectors[i], employer_vectors[j]);
            std::cout << "- Employer " << j + 1 << " (Similarity: " << similarity << ")\n";
        }
        std::cout << std::endl;
    }

    return 0;
}
