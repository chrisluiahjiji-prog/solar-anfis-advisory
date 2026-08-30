function generateSolarModel(lat, lon, locationName)
% GENERATESOLARMODEL Trains an ANFIS solar advisory model for any location
% Usage: generateSolarModel(10.07, 76.68, "Kochi")

fprintf('\n=== Generating solar model for %s (%.2f, %.2f) ===\n', locationName, lat, lon);

%% 1. DOWNLOAD DATA DIRECTLY FROM NASA POWER API
startDate = '20200101';
endDate = '20260719';
url = sprintf(['https://power.larc.nasa.gov/api/temporal/daily/point?' ...
    'parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M&community=RE&' ...
    'longitude=%f&latitude=%f&start=%s&end=%s&format=JSON'], ...
    lon, lat, startDate, endDate);

fprintf('Downloading NASA POWER data...\n');
options = weboptions('Timeout', 60);
raw = webread(url, options);

params = raw.properties.parameter;
dateKeys = fieldnames(params.ALLSKY_SFC_SW_DWN);
n = length(dateKeys);

YEAR = zeros(n,1); MO = zeros(n,1); DY = zeros(n,1);
ALLSKY_SFC_SW_DWN = zeros(n,1); T2M = zeros(n,1); RH2M = zeros(n,1); WS10M = zeros(n,1);

for i = 1:n
    dk = dateKeys{i};
    YEAR(i) = str2double(dk(2:5));
    MO(i)   = str2double(dk(6:7));
    DY(i)   = str2double(dk(8:9));
    ALLSKY_SFC_SW_DWN(i) = params.ALLSKY_SFC_SW_DWN.(dk);
    T2M(i)  = params.T2M.(dk);
    RH2M(i) = params.RH2M.(dk);
    WS10M(i)= params.WS10M.(dk);
end

data = table(YEAR, MO, DY, ALLSKY_SFC_SW_DWN, T2M, RH2M, WS10M);
fprintf('Downloaded %d rows.\n', height(data));

%% 2. CLEAN -999 / MISSING
data.T2M(data.T2M == -999) = NaN;
data.RH2M(data.RH2M == -999) = NaN;
data.WS10M(data.WS10M == -999) = NaN;
data.ALLSKY_SFC_SW_DWN(data.ALLSKY_SFC_SW_DWN == -999) = NaN;
data = rmmissing(data);
fprintf('Rows after cleaning: %d\n', height(data));

%% 3. SORT + SPLIT
data = sortrows(data, {'YEAR','MO','DY'});
n = height(data);
cutoff = round(0.8 * n);
trainData = data(1:cutoff, :);
testData  = data(cutoff+1:end, :);

trainX = trainData(:, {'T2M','RH2M','WS10M'});
trainY = trainData.ALLSKY_SFC_SW_DWN;
testX  = testData(:, {'T2M','RH2M','WS10M'});
testY  = testData.ALLSKY_SFC_SW_DWN;

%% 4. TRAIN ANFIS
trainMatrix = [table2array(trainX), trainY];
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = 3;
opt.InputMembershipFunctionType = "gaussmf";
initialFIS = genfis(table2array(trainX), trainY, opt);

anfisOpt = anfisOptions('InitialFIS', initialFIS);
anfisOpt.EpochNumber = 50;
anfisOpt.DisplayANFISInformation = 0;
anfisOpt.DisplayErrorValues = 0;
[trainedFIS, ~] = anfis(trainMatrix, anfisOpt);

%% 5. TEST + CLIP
predictedY = evalfis(trainedFIS, table2array(testX));
predictedY = max(predictedY, 0);
predictedY = min(predictedY, max(trainY) * 1.2);
rmseANFIS = sqrt(mean((testY - predictedY).^2));
maeANFIS  = mean(abs(testY - predictedY));
fprintf('ANFIS RMSE: %.4f | MAE: %.4f\n', rmseANFIS, maeANFIS);
% Save the trained model for reuse later (e.g., bridging recent data gaps)
save(sprintf('model_%s.mat', locationName), 'trainedFIS', 'trainY');

%% 6. CONFIDENCE FLAG (monsoon = Jun-Sep, same for all Kerala; adjust if needed elsewhere)
isMonsoon = ismember(testData.MO, [6 7 8 9]);
windowSize = 7;
rollingStd = zeros(height(testData), 1);
for i = 1:height(testData)
    startIdx = max(1, i - windowSize + 1);
    rollingStd(i) = std(testY(startIdx:i));
end
varThreshold = prctile(rollingStd, 75);
confidenceFlag = strings(height(testData), 1);
for i = 1:height(testData)
    if isMonsoon(i) || rollingStd(i) > varThreshold
        confidenceFlag(i) = "Low confidence";
    else
        confidenceFlag(i) = "High confidence";
    end
end

%% 7. EXPLANATIONS
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

%% 8. EXPORT HISTORICAL RESULTS
exportTable = table(testData.YEAR, testData.MO, testData.DY, ...
    testData.T2M, testData.RH2M, testData.WS10M, ...
    testY, predictedY, confidenceFlag, explanationText, ...
    'VariableNames', {'Year','Month','Day','Temperature','Humidity','WindSpeed', ...
                       'ActualIrradiance','PredictedIrradiance','Confidence','Explanation'});
histFilename = sprintf('dashboard_data_%s.csv', locationName);
writetable(exportTable, histFilename);
fprintf('Exported %s\n', histFilename);

%% 9. LIVE 16-DAY FORECAST
fUrl = sprintf(['https://api.open-meteo.com/v1/forecast?latitude=%f&longitude=%f&' ...
    'daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean&' ...
    'forecast_days=16&timezone=Asia%%2FKolkata&wind_speed_unit=ms'], lat, lon);
fData = webread(fUrl, options);
fDates = fData.daily.time;
fTemp = fData.daily.temperature_2m_mean;
fHum  = fData.daily.relative_humidity_2m_mean;
fWind = fData.daily.wind_speed_10m_mean;

nF = length(fDates);
fX = [fTemp(:), fHum(:), fWind(:)];
fPred = evalfis(trainedFIS, fX);
fPred = max(fPred, 0);
fPred = min(fPred, max(trainY) * 1.2);

fConf = strings(nF,1);
fExpl = strings(nF,1);
for k = 1:nF
    m = month(datetime(fDates{k}));
    isM = ismember(m, [6 7 8 9]);
    fConf(k) = "High confidence";
    if isM || k > 10 % beyond 10 days: automatically lower confidence (forecast uncertainty)
        fConf(k) = "Low confidence";
    end

    inputVals = fX(k,:);
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
    if fPred(k) < median(trainY)
        outcome = 'Lower-than-average output';
    else
        outcome = 'Higher-than-average output';
    end
    if isempty(reasons)
        fExpl(k) = sprintf('%s is mainly due to typical mid-range weather conditions.', outcome);
    else
        fExpl(k) = sprintf('%s is mainly due to %s.', outcome, strjoin(reasons, ' and '));
    end
end

forecastTable = table(fDates, fTemp, fHum, fWind, fPred, fConf, fExpl, ...
    'VariableNames', {'Date','Temp','Humidity','Wind','PredictedIrradiance','Confidence','Explanation'});
foreFilename = sprintf('forecast_data_%s.csv', locationName);
writetable(forecastTable, foreFilename);
fprintf('Exported %s\n', foreFilename);

fprintf('=== Done: %s ===\n\n', locationName);
end