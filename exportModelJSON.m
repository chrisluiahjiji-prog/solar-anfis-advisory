function exportModelJSON(locationName)
% EXPORTMODELJSON Exports a trained ANFIS model's fuzzy rules and
% membership functions to a JSON file the dashboard can read and
% evaluate directly in JavaScript — no MATLAB needed at runtime.
% Usage: exportModelJSON("Kochi")   (after loading its saved model)

loaded = load(sprintf('model_%s.mat', locationName));
fis = loaded.trainedFIS;
trainY = loaded.trainY;

model = struct();
model.location = locationName;
model.maxTrainY = max(trainY) * 1.2;

% --- Input ranges + membership functions ---
model.inputs = {};
for i = 1:3
    inp = struct();
    inp.name = fis.Inputs(i).Name;
    inp.range = fis.Inputs(i).Range;
    mfs = {};
    for j = 1:length(fis.Inputs(i).MembershipFunctions)
        mf = fis.Inputs(i).MembershipFunctions(j);
        mfs{end+1} = struct('type', mf.Type, 'params', mf.Parameters); %#ok<AGROW>
    end
    inp.mfs = mfs;
    model.inputs{end+1} = inp; %#ok<AGROW>
end

% --- Rules: antecedent MF indices per input + consequent linear params ---
model.rules = {};
numRules = length(fis.Rules);
for r = 1:numRules
    rule = struct();
    ant = fis.Rules(r).Antecedent; % vector of MF index per input (1-based)
    rule.antecedent = ant;
    outMF = fis.Outputs(1).MembershipFunctions(r);
    rule.consequent = outMF.Parameters; % [p1 p2 p3 constant]
    model.rules{end+1} = rule; %#ok<AGROW>
end

jsonStr = jsonencode(model);
fname = sprintf('model_%s.json', locationName);
fid = fopen(fname, 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);
fprintf('Exported %s (%d rules)\n', fname, numRules);
end