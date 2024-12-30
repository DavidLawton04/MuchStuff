# include <iostream>
# include <cmath>
# include <random>
# include <list>
# include <vector>

using namespace std;

// Newton's Method

double sqrt(double n) {

    double root = n / 2;
    double *root_ptr = &root;

    for (int i=0; i<10; i++) {
        root = 0.5 * (root+n/root);
    }

    cout << "Root of " << n << " is " << root << endl;
    
    return root;
}

string random_quote(int quote_length, string alphabet, string quote) {
    string monkey_quote = "";
    string rand_letter;

    int alphabet_length = alphabet.length();

    for (int i=0; i<quote_length; i++) {
        rand_letter = alphabet[rand() % alphabet_length];
        monkey_quote += rand_letter;
    }

    return monkey_quote;
}


vector<int> maximum_of_pairs(vector<int> score_vals) {
    vector<int> max_score_vals;
    for (int i=0; i<score_vals.size()/2; i++) {
        max_score_vals.push_back(max(score_vals[2*i], score_vals[2*i+1]));
    }
    return max_score_vals;
}


void infinite_monkey(string quote = "methinks") {

    string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,. ";
    int alphabet_length = alphabet.length();
    int quote_length = quote.length();
    int N = 10000;

    string monkey_quote = "";
    int count = 0;
    vector<int> scores;
    float score;

    while ((monkey_quote != quote) && (count < N)) {
        score = 0;
        monkey_quote = random_quote(quote_length, alphabet, quote);
        for (int i=0;i<quote_length;i++) {
            if (quote[i] == monkey_quote[i]) {
                score += 1;
            }
        }
        scores.push_back(score);
        count += 1;
    }
    cout <<  "No. of scores = " << scores.size() << endl;

    while (scores.size() > 1) {
        scores = maximum_of_pairs(scores);
    }

    int max_score = max(scores[0], scores[1]);
    cout << "Max score: " << max_score << endl;
    float highest_percent = max_score / quote_length * 100;

    if (count == N) {
        cout << "Monkey has not typed the quote: \"" << quote << "\" in " << N <<" attempts, " << " highest percentage was " << highest_percent << endl;
    } else {
        cout << "Monkey has typed the quote: \"" << monkey_quote << "\" in " << count << " attempts" << endl;
        cout << monkey_quote << endl;
    }
}



int main() {

    double n;
    cout << "Enter a number: ";
    cin >> n;

    sqrt(n);

    infinite_monkey();

    return 0;
}

