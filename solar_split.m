%% ================= SOLAR ANFIS PROJECT — FULL PIPELINE =================

%% 1. LOAD DATA
data = readtable('Solar project.csv');

%% 2. CLEAN -999 / MISSING VALUES
data.T2M(data.T2M == -999) = NaN;
data.RH2M(data.RH2M == -999) = NaN;
data.WS10M(data.WS10M == -999) = NaN;
data.ALLSKY_SFC_SW_DWN(data.ALLSKY_SFC_SW_DWN == -999) = NaN;
data = rmmissing(data);

disp('Rows after removing -999/missing values:');
disp(height(data));

%% 3. SORT CHRONOLOGICALLY
data = sortrows(data, {'YEAR','MO','DY'});

%% 4. CHRONOLOGICAL 80/20 SPLIT
n = height(data);
cutoff = round(0.8 * n);
trainData = data(1:cutoff, :);
testData  = data(cutoff+1:end, :);

disp('Train size:'); disp(size(trainData));
disp('Test size:');  disp(size(testData));

%% 5. BUILD X (INPUTS) AND Y (OUTPUT)
trainX = trainData(:, {'T2M','RH2M','WS10M'});
trainY = trainData.ALLSKY_SFC_SW_DWN;
testX  = testData(:, {'T2M','RH2M','WS10M'});
testY  = testData.ALLSKY_SFC_SW_DWN;

%% 6. TRAIN ANFIS
trainMatrix = [table2array(trainX), trainY];

opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = 3;
opt.InputMembershipFunctionType = "gaussmf";
initialFIS = genfis(table2array(trainX), trainY, opt);

anfisOpt = anfisOptions('InitialFIS', initialFIS);
anfisOpt.EpochNumber = 50;
anfisOpt.DisplayANFISInformation = 0;
anfisOpt.DisplayErrorValues = 1;

[trainedFIS, trainError] = anfis(trainMatrix, anfisOpt);

%% 7. TEST ANFIS  (**FIX ADDED HERE**)
predictedY = evalfis(trainedFIS, table2array(testX));

% Clip predictions to physically sensible bounds (irradiance can't be negative
% or absurdly higher than anything seen in training)
predictedY = max(predictedY, 0);
predictedY = min(predictedY, max(trainY) * 1.2);

rmseANFIS = sqrt(mean((testY - predictedY).^2));
maeANFIS  = mean(abs(testY - predictedY));
fprintf('ANFIS RMSE: %.4f\n', rmseANFIS);
fprintf('ANFIS MAE: %.4f\n', maeANFIS);

figure;
plot(testY, 'b'); hold on;
plot(predictedY, 'r--');
legend('Actual Irradiance', 'ANFIS Predicted');
xlabel('Test Day Index'); ylabel('Irradiance (kWh/m^2/day)');
title('ANFIS Prediction vs Actual');

%% 8. BASELINE 1 — NAIVE (tomorrow = today)
naivePred = [testY(1); testY(1:end-1)];
rmseNaive = sqrt(mean((testY - naivePred).^2));
maeNaive  = mean(abs(testY - naivePred));
fprintf('Naive Baseline RMSE: %.4f\n', rmseNaive);
fprintf('Naive Baseline MAE: %.4f\n', maeNaive);

%% 9. BASELINE 2 — LINEAR REGRESSION
lm = fitlm(trainX, trainY);
linPred = predict(lm, testX);
rmseLin = sqrt(mean((testY - linPred).^2));
maeLin  = mean(abs(testY - linPred));
fprintf('Linear Regression RMSE: %.4f\n', rmseLin);
fprintf('Linear Regression MAE: %.4f\n', maeLin);

%% 10. COMPARISON SUMMARY
fprintf('\n--- MODEL COMPARISON ---\n');
fprintf('%-20s %-10s %-10s\n', 'Model', 'RMSE', 'MAE');
fprintf('%-20s %-10.4f %-10.4f\n', 'ANFIS', rmseANFIS, maeANFIS);
fprintf('%-20s %-10.4f %-10.4f\n', 'Naive', rmseNaive, maeNaive);
fprintf('%-20s %-10.4f %-10.4f\n', 'Linear Regression', rmseLin, maeLin);

%% 11. MONSOON-AWARE CONFIDENCE FLAG
% Kerala monsoon: roughly June (6) through September (9)
isMonsoon = ismember(testData.MO, [6 7 8 9]);

% Recent variability: rolling 7-day std dev of actual irradiance up to each test day
windowSize = 7;
rollingStd = zeros(height(testData), 1);
for i = 1:height(testData)
    startIdx = max(1, i - windowSize + 1);
    rollingStd(i) = std(testY(startIdx:i));
end

% Threshold: "high variability" = top 25% of rolling std dev values
varThreshold = prctile(rollingStd, 75);

% Confidence flag: Low if monsoon OR high recent variability, else High
confidenceFlag = strings(height(testData), 1);
for i = 1:height(testData)
    if isMonsoon(i) || rollingStd(i) > varThreshold
        confidenceFlag(i) = "Low confidence";
    else
        confidenceFlag(i) = "High confidence";
    end
end

% Summary counts
disp('Confidence flag distribution:');
disp(countcats(categorical(confidenceFlag)));
disp(categories(categorical(confidenceFlag)));

% Example: show first 10 days with their flag
fullTable = table(testData.YEAR, testData.MO, testData.DY, ...
    testY, predictedY, confidenceFlag, ...
    'VariableNames', {'Year','Month','Day','Actual','Predicted','Confidence'});

disp(fullTable);


%% 12. BEST-DAY RECOMMENDATION FOR HIGH-POWER APPLIANCE USE
todayIdx = 150; % pick any test-day index as "today" — change this to try different weeks
horizon = 7;

if todayIdx + horizon - 1 <= height(testData)
    weekDates = [testData.YEAR(todayIdx:todayIdx+horizon-1), ...
        testData.MO(todayIdx:todayIdx+horizon-1), ...
        testData.DY(todayIdx:todayIdx+horizon-1)];
    weekPredicted = predictedY(todayIdx:todayIdx+horizon-1);
    weekConfidence = confidenceFlag(todayIdx:todayIdx+horizon-1);

    weekTable = table(weekDates(:,1), weekDates(:,2), weekDates(:,3), ...
        weekPredicted, weekConfidence, ...
        'VariableNames', {'Year','Month','Day','PredictedIrradiance','Confidence'});

    disp('7-day forecast window:');
    disp(weekTable);

    [bestVal, bestRelIdx] = max(weekPredicted);
    bestDate = weekDates(bestRelIdx, :);

    fprintf('\nBest day to run high-power appliances: %d/%d/%d (predicted irradiance: %.2f kWh/m^2/day, %s)\n', ...
        bestDate(1), bestDate(2), bestDate(3), bestVal, weekConfidence(bestRelIdx));
else
    disp('Not enough remaining test days for a 7-day window at this index — pick a smaller todayIdx.');
end

%% 13. LOCAL "SOLAR DAY" STREAK TRACKER
% Use the FULL cleaned dataset (not just test set) for richer historical patterns
threshold = prctile(data.ALLSKY_SFC_SW_DWN, 75); % "high-irradiance day" = top 25%
fprintf('High-irradiance threshold (75th percentile): %.2f kWh/m^2/day\n', threshold);

isHighDay = data.ALLSKY_SFC_SW_DWN >= threshold;

% --- Find longest streak of consecutive high-irradiance days, per year ---
years = unique(data.YEAR);
fprintf('\nLongest high-irradiance streak per year:\n');
for y = years'
    yearMask = data.YEAR == y;
    yearHighDays = isHighDay(yearMask);

    % Count longest run of consecutive 1s
    maxStreak = 0; currentStreak = 0;
    for i = 1:length(yearHighDays)
        if yearHighDays(i)
            currentStreak = currentStreak + 1;
            maxStreak = max(maxStreak, currentStreak);
        else
            currentStreak = 0;
        end
    end
    fprintf('  %d: %d days\n', y, maxStreak);
end

% --- Compare a given month to the same month in previous years ---
targetMonth = 7; % July, for example
fprintf('\nAverage irradiance in month %d, by year:\n', targetMonth);
for y = years'
    monthMask = data.YEAR == y & data.MO == targetMonth;
    if any(monthMask)
        avgIrr = mean(data.ALLSKY_SFC_SW_DWN(monthMask));
        fprintf('  %d: %.2f kWh/m^2/day (avg)\n', y, avgIrr);
    end
end


%% 14. EXPLAINABILITY LAYER — PLAIN-ENGLISH RULE EXTRACTION
% Pick a specific test day to explain (e.g., the worst-performing week from before, or any index)
explainIdx = 100; % change this to explain a different day

inputVals = table2array(testX(explainIdx, :)); % [T2M, RH2M, WS10M]

% Get membership function labels for each input at this day's values
inputNames = {'Temperature', 'Humidity', 'Wind Speed'};
labels = {'Low','Medium','High'}; % since we used 3 gaussmf per input

fprintf('\n--- Explaining prediction for %d/%d/%d ---\n', ...
    testData.YEAR(explainIdx), testData.MO(explainIdx), testData.DY(explainIdx));
fprintf('Predicted irradiance: %.2f kWh/m^2/day\n\n', predictedY(explainIdx));

% For each input, find which membership function fires strongest
for i = 1:3
    mfDegrees = zeros(1,3);
    for j = 1:3
        mf = trainedFIS.Inputs(i).MembershipFunctions(j);
        mfDegrees(j) = evalmf(mf, inputVals(i));
    end
    [~, bestMF] = max(mfDegrees);
    fprintf('%s = %.2f  -->  mostly "%s"\n', inputNames{i}, inputVals(i), labels{bestMF});
end

% Simple plain-English sentence generator based on dominant conditions
fprintf('\nPlain-English explanation:\n');
[~, tempMF] = max(arrayfun(@(j) evalmf(trainedFIS.Inputs(1).MembershipFunctions(j), inputVals(1)), 1:3));
[~, humMF]  = max(arrayfun(@(j) evalmf(trainedFIS.Inputs(2).MembershipFunctions(j), inputVals(2)), 1:3));
[~, windMF] = max(arrayfun(@(j) evalmf(trainedFIS.Inputs(3).MembershipFunctions(j), inputVals(3)), 1:3));

reasons = {};
if humMF == 3
    reasons{end+1} = 'high humidity';
elseif humMF == 1
    reasons{end+1} = 'low humidity';
end
if windMF == 1
    reasons{end+1} = 'low wind';
elseif windMF == 3
    reasons{end+1} = 'strong wind';
end
if tempMF == 3
    reasons{end+1} = 'high temperature';
elseif tempMF == 1
    reasons{end+1} = 'cooler temperature';
end

if predictedY(explainIdx) < median(trainY)
    outcome = 'lower-than-average output';
else
    outcome = 'higher-than-average output';
end

if isempty(reasons)
    fprintf('Today''s %s is mainly due to typical mid-range weather conditions.\n', outcome);
else
    fprintf('Today''s %s is mainly due to %s.\n', outcome, strjoin(reasons, ' and '));
end


%% 15. COMBINED DAILY SOLAR ADVISORY OUTPUT
function printSolarAdvisory(dayIdx, testData, predictedY, confidenceFlag, trainedFIS, testX, trainY)

fprintf('\n========================================\n');
fprintf(' SOLAR ADVISORY REPORT — %d/%d/%d\n', ...
    testData.YEAR(dayIdx), testData.MO(dayIdx), testData.DY(dayIdx));
fprintf('========================================\n');

fprintf('Predicted Irradiance: %.2f kWh/m^2/day\n', predictedY(dayIdx));
fprintf('Confidence Level:     %s\n', confidenceFlag(dayIdx));

% --- Explanation ---
inputVals = table2array(testX(dayIdx, :));
inputNames = {'Temperature', 'Humidity', 'Wind Speed'};
labels = {'Low','Medium','High'};

fprintf('\nWeather Conditions:\n');
mfChoice = zeros(1,3);
for i = 1:3
    mfDegrees = zeros(1,3);
    for j = 1:3
        mf = trainedFIS.Inputs(i).MembershipFunctions(j);
        mfDegrees(j) = evalmf(mf, inputVals(i));
    end
    [~, mfChoice(i)] = max(mfDegrees);
    fprintf('  %-12s %.2f  (%s)\n', inputNames{i}, inputVals(i), labels{mfChoice(i)});
end

reasons = {};
if mfChoice(2) == 3, reasons{end+1} = 'high humidity'; 
elseif mfChoice(2) == 1, reasons{end+1} = 'low humidity'; end
if mfChoice(3) == 1, reasons{end+1} = 'low wind'; 
elseif mfChoice(3) == 3, reasons{end+1} = 'strong wind'; end
if mfChoice(1) == 3, reasons{end+1} = 'high temperature'; 
elseif mfChoice(1) == 1, reasons{end+1} = 'cooler temperature'; end

if predictedY(dayIdx) < median(trainY)
    outcome = 'Lower-than-average output';
else
    outcome = 'Higher-than-average output';
end

fprintf('\nExplanation: ');
if isempty(reasons)
    fprintf('%s is mainly due to typical mid-range weather conditions.\n', outcome);
else
    fprintf('%s is mainly due to %s.\n', outcome, strjoin(reasons, ' and '));
end

fprintf('========================================\n');
end

% --- Call it for a chosen day ---
printSolarAdvisory(20, testData, predictedY, confidenceFlag, trainedFIS, testX, trainY);

%% 16. EXPORT RESULTS FOR WEB DASHBOARD

% Generate explanation for every test day (not just the ones we manually checked)
explanationText = strings(height(testData), 1);
for k = 1:height(testData)
    inputVals = table2array(testX(k, :));
    mfChoice = zeros(1,3);
    for i = 1:3
        mfDegrees = zeros(1,3);
        for j = 1:3
            mf = trainedFIS.Inputs(i).MembershipFunctions(j);
            mfDegrees(j) = evalmf(mf, inputVals(i));
        end
        [~, mfChoice(i)] = max(mfDegrees);
    end
    reasons = {};
    if mfChoice(2) == 3, reasons{end+1} = 'high humidity';
    elseif mfChoice(2) == 1, reasons{end+1} = 'low humidity'; end
    if mfChoice(3) == 1, reasons{end+1} = 'low wind';
    elseif mfChoice(3) == 3, reasons{end+1} = 'strong wind'; end
    if mfChoice(1) == 3, reasons{end+1} = 'high temperature';
    elseif mfChoice(1) == 1, reasons{end+1} = 'cooler temperature'; end

    if predictedY(k) < median(trainY)
        outcome = 'Lower-than-average output';
    else
        outcome = 'Higher-than-average output';
    end

    if isempty(reasons)
        explanationText(k) = sprintf('%s is mainly due to typical mid-range weather conditions.', outcome);
    else
        explanationText(k) = sprintf('%s is mainly due to %s.', outcome, strjoin(reasons, ' and '));
    end
end

% Build the final export table
exportTable = table(testData.YEAR, testData.MO, testData.DY, ...
    testData.T2M, testData.RH2M, testData.WS10M, ...
    testY, predictedY, confidenceFlag, explanationText, ...
    'VariableNames', {'Year','Month','Day','Temperature','Humidity','WindSpeed', ...
    'ActualIrradiance','PredictedIrradiance','Confidence','Explanation'});

writetable(exportTable, 'dashboard_data.csv');
disp('Exported dashboard_data.csv successfully.');
disp(height(exportTable));

url = "https://api.open-meteo.com/v1/forecast?latitude=10.07&longitude=76.68&daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean&forecast_days=16&timezone=Asia%2FKolkata&wind_speed_unit=ms";
options = weboptions('Timeout', 15);
forecastData = webread(url, options);

forecastDates = forecastData.daily.time;
tempF = forecastData.daily.temperature_2m_mean;
humF  = forecastData.daily.relative_humidity_2m_mean;
windF = forecastData.daily.wind_speed_10m_mean;

futureX = [tempF(:), humF(:), windF(:)];
futurePredicted = evalfis(trainedFIS, futureX);
futurePredicted = max(futurePredicted, 0);
futurePredicted = min(futurePredicted, max(trainY) * 1.2);

disp(table(forecastDates, tempF, humF, windF, futurePredicted, ...
    'VariableNames', {'Date','Temp','Humidity','Wind','PredictedIrradiance'}));

% Confidence flag for forecast days (same monsoon-aware logic as before)
isMonsoonF = ismember(month(datetime(forecastDates)), [6 7 8 9]);
forecastConfidence = strings(7,1);
for i = 1:7
    if isMonsoonF(i)
        forecastConfidence(i) = "Low confidence";
    else
        forecastConfidence(i) = "High confidence";
    end
end

% Explanation for each forecast day
forecastExplanation = strings(7,1);
for k = 1:7
    inputVals = futureX(k,:);
    mfChoice = zeros(1,3);
    for i = 1:3
        mfDegrees = zeros(1,3);
        for j = 1:3
            mf = trainedFIS.Inputs(i).MembershipFunctions(j);
            mfDegrees(j) = evalmf(mf, inputVals(i));
        end
        [~, mfChoice(i)] = max(mfDegrees);
    end
    reasons = {};
    if mfChoice(2) == 3, reasons{end+1} = 'high humidity';
    elseif mfChoice(2) == 1, reasons{end+1} = 'low humidity'; end
    if mfChoice(3) == 1, reasons{end+1} = 'low wind';
    elseif mfChoice(3) == 3, reasons{end+1} = 'strong wind'; end
    if mfChoice(1) == 3, reasons{end+1} = 'high temperature';
    elseif mfChoice(1) == 1, reasons{end+1} = 'cooler temperature'; end

    if futurePredicted(k) < median(trainY)
        outcome = 'Lower-than-average output';
    else
        outcome = 'Higher-than-average output';
    end
    if isempty(reasons)
        forecastExplanation(k) = sprintf('%s is mainly due to typical mid-range weather conditions.', outcome);
    else
        forecastExplanation(k) = sprintf('%s is mainly due to %s.', outcome, strjoin(reasons, ' and '));
    end
end

forecastTable = table(forecastDates, tempF, humF, windF, futurePredicted, forecastConfidence, forecastExplanation, ...
    'VariableNames', {'Date','Temp','Humidity','Wind','PredictedIrradiance','Confidence','Explanation'});

disp(forecastTable);
writetable(forecastTable, 'forecast_data.csv');
disp('Exported forecast_data.csv');