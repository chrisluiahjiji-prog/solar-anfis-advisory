function generateBridgeData(lat, lon, locationName, gapStartDate)
% GENERATEBRIDGEDATA Fills the gap between historical dataset end and today
% using Open-Meteo's historical archive, run through the saved trained model.
% Usage: generateBridgeData(10.07, 76.68, "Kochi", "2026-07-20")

fprintf('\n=== Bridging recent gap for %s ===\n', locationName);

% Load the saved trained model for this location
loaded = load(sprintf('model_%s.mat', locationName));
trainedFIS = loaded.trainedFIS;
trainY = loaded.trainY;

% End date = yesterday (archive data typically lags a few days behind today)
endDate = datestr(datetime('now') - days(2), 'yyyy-mm-dd');

url = sprintf(['https://archive-api.open-meteo.com/v1/archive?latitude=%f&longitude=%f&' ...
    'start_date=%s&end_date=%s&' ...
    'daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean&' ...
    'wind_speed_unit=ms&timezone=Asia%%2FKolkata'], lat, lon, gapStartDate, endDate);

options = weboptions('Timeout', 30);
archiveData = webread(url, options);

dates = archiveData.daily.time;
tempB = archiveData.daily.temperature_2m_mean;
humB  = archiveData.daily.relative_humidity_2m_mean;
windB = archiveData.daily.wind_speed_10m_mean;

nB = length(dates);
fprintf('Retrieved %d bridge days (%s to %s)\n', nB, gapStartDate, endDate);

bX = [tempB(:), humB(:), windB(:)];
bPred = evalfis(trainedFIS, bX);
bPred = max(bPred, 0);
bPred = min(bPred, max(trainY) * 1.2);

bConf = strings(nB,1);
bExpl = strings(nB,1);
for k = 1:nB
    m = month(datetime(dates{k}));
    isM = ismember(m, [6 7 8 9]);
    bConf(k) = "High confidence";
    if isM
        bConf(k) = "Low confidence";
    end

    inputVals = bX(k,:);
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
    if bPred(k) < median(trainY)
        outcome = 'Lower-than-average output';
    else
        outcome = 'Higher-than-average output';
    end
    if isempty(reasons)
        bExpl(k) = sprintf('%s is mainly due to typical mid-range weather conditions.', outcome);
    else
        bExpl(k) = sprintf('%s is mainly due to %s.', outcome, strjoin(reasons, ' and '));
    end
end

bridgeTable = table(dates, tempB, humB, windB, bPred, bConf, bExpl, ...
    'VariableNames', {'Date','Temp','Humidity','Wind','PredictedIrradiance','Confidence','Explanation'});
fname = sprintf('bridge_data_%s.csv', locationName);
writetable(bridgeTable, fname);
fprintf('Exported %s\n', fname);
end