clc, clear all, close all, warning off

%% Load Data
fileName = "NFL scores.txt";
fileID = fopen(fileName);
C = textscan(fileID, '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s');
fclose(fileID);

% team names
team_names = C{1}(3 : end);

% weekly score
week_scores = zeros(length(team_names), length(C) - 1);
for i = 1 : length(C) - 1
    weekly = C{i + 1}(3 : end);
    for j = 1 : length(weekly)
        week_scores(j, i) = str2num(weekly{j});
    end
end

% plot graph
figure
plot(week_scores, '.-')
xlabel("Week")
ylabel("Teams")
title("Weekly Scores of Teams")

%% Standardize Data
mx = max(week_scores')';
mn = min(week_scores')';

% mu = mean(week_scores')';
% sig = std(week_scores')';
mu = zeros(32, 1);
sig = ones(32, 1);

standardized_weekly_scores = (week_scores - mu) ./ sig;

%% Prepare Predictors and Responses
XTrain = standardized_weekly_scores(:, 1 : end - 1);
YTrain = standardized_weekly_scores(:, 2 : end);

%% Define LSTM Network Architecture
numFeatures = size(XTrain, 1);
numResponses = size(YTrain, 1);
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

%% Train LSTM Network
net = trainNetwork(XTrain, YTrain, layers, options);

%% Forecast Future Time Steps
net = predictAndUpdateState(net, XTrain);
[net, YPred] = predictAndUpdateState(net, YTrain(:, end));

numTimeStepsTest = size(XTrain, 2);
for i = 2 : numTimeStepsTest + 2
    [net, YPred(:, i)] = predictAndUpdateState(net, YPred(:, i - 1), 'ExecutionEnvironment', 'cpu');
end

YPred = sig .* YPred + mu;
YPred = round(YPred(:, 2 : end));

ymn = min(YPred')';
ymx = max(YPred')';

YPred = round((((YPred - ymn) ./ (ymx - ymn)) .* (mx - mn)) + mn);

figure
plot(YPred, '.-')
xlabel("2020 Week")
ylabel("Teams")
title("2020 Forecast")

%% Save Results
T = table(YPred, 'RowNames', team_names);
writetable(T, '2020.txt', 'WriteRowNames', true) 