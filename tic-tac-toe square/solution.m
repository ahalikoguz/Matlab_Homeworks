clc;
clear;

%% Preprocess the data
% In the train file, 'x' is set to 2, 'o' to 3, and 'b' to 4.
% Negative is set to 0 and positive to 1.
% The file is read and the data is stored in matrix A.
A = dlmread('train.txt');

[x,y] = size(A);

% Matrix S stores the last column of the train set, which contains the results.
S = A(:,y);

% Counters for the number of 1's and 0's in the result matrix S.
count_result1 = 0;
count_result0 = 0;

% Counting the total number of 1's and 0's in matrix S.
for m = 1:x
    if S(m) == 1 
        count_result1 = count_result1 + 1;
    end
    if S(m) == 0
        count_result0 = count_result0 + 1;
    end
end

%% Calculate probabilities
% Matrix Y stores the probabilities.
% Rows 1 and 2 for 'x' with positive=1 and negative=0 outcomes.
% Rows 3 and 4 for 'o' with positive=1 and negative=0 outcomes.
% Rows 5 and 6 for 'b' with positive=1 and negative=0 outcomes.
Y = zeros(6, y-1);

% Matrix O stores the final probabilities.
O = zeros(6, y-1);

for n = 1:y-1
    X = A(:,n);
    for j = 1:x
        % Populate matrix Y with counts for each condition.
        if X(j) == 2 && S(j) == 1
            Y(1, n) = Y(1, n) + 1;
        elseif X(j) == 2 && S(j) == 0
            Y(2, n) = Y(2, n) + 1;
        elseif X(j) == 3 && S(j) == 1
            Y(3, n) = Y(3, n) + 1;
        elseif X(j) == 3 && S(j) == 0
            Y(4, n) = Y(4, n) + 1;
        elseif X(j) == 4 && S(j) == 1
            Y(5, n) = Y(5, n) + 1;
        elseif X(j) == 4 && S(j) == 0
            Y(6, n) = Y(6, n) + 1;
        end
    end

    % Calculate the probability values and store in matrix O.
    O(1, n) = Y(1, n) / count_result1;
    O(2, n) = Y(2, n) / count_result0;
    O(3, n) = Y(3, n) / count_result1;
    O(4, n) = Y(4, n) / count_result0;
    O(5, n) = Y(5, n) / count_result1;
    O(6, n) = Y(6, n) / count_result0;
end

%%
% Calculate probabilities for the test data.
F = dlmread('test.txt');

[k, l] = size(F);

% Matrix T stores probabilities for each row in test data: 
% probability of positive outcome, probability of negative outcome,
% expected result, actual result, and accuracy comparison.
T = zeros(k, 5);

% Matrices Z and W store cumulative probabilities for positive and negative outcomes, respectively.
Z = ones(k, l);
W = ones(k, l);

for f = 1:k
    for i = 1:l-1
        % Extract corresponding probabilities for each condition from F matrix.
        if F(f, i) == 2
            Z(f, i) = O(1, i);
            W(f, i) = O(2, i); 
        elseif F(f, i) == 3
            Z(f, i) = O(3, i);
            W(f, i) = O(4, i); 
        elseif F(f, i) == 4
            Z(f, i) = O(5, i);
            W(f, i) = O(6, i); 
        end

        % Multiply the probability values for each condition.
        Z(f, l) = Z(f, i) * Z(f, l);
        W(f, l) = W(f, i) * W(f, l);
    end

    % Calculate final probabilities for positive and negative outcomes.
    P1 = Z(f, l) * (count_result1 / (count_result1 + count_result0));
    P0 = W(f, l) * (count_result0 / (count_result1 + count_result0));

    Probability1 = (P1 / (P1 + P0));
    Probability0 = (P0 / (P1 + P0));

    % Store the calculated probabilities and expected outcomes in matrix T.
    T(f, 1) = Probability1;
    T(f, 2) = Probability0;
    T(f, 3) = (Probability1 >= Probability0) ? 1 : 0;
    T(f, 4) = F(f, l);
    T(f, 5) = (T(f, 3) == T(f, 4)) ? 1 : 0;
end

% Count the number of correct and incorrect predictions in matrix T.
correct_count = sum(T(:, 5) == 1);
incorrect_count = sum(T(:, 5) == 0);

% Calculate the overall accuracy of the predictions.
accuracy = correct_count / (correct_count + incorrect_count);

