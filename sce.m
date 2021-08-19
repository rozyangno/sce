function Y = sce(P, maxIter, alpha, weights, Y0, eta0, bConstantEta, nRepuSamp, blockSize, blockCount)
nn = size(P,1);
if ~exist('weights', 'var') || isempty(weights)
    weights = ones(nn,1);
end
if ~exist('alpha', 'var') || isempty(alpha)
    alpha = 0.5;
end
if ~exist('Y0', 'var') || isempty(Y0)
    rng(0, 'twister');
    Y0 = randn(nn,2)*1e-4;
end
if ~exist('maxIter', 'var') || isempty(maxIter)
    maxIter = 1e5;
end
if ~exist('eta0', 'var') || isempty(eta0)
    eta0 = 1;
end
if ~exist('nRepuSamp', 'var') || isempty(nRepuSamp)
    nRepuSamp = 1;
end
if ~exist('blockSize', 'var') || isempty(blockSize)
    blockSize = 128;
end
if ~exist('blockCount', 'var') || isempty(blockCount)
    blockCount = 128;
end
if ~exist('bConstantEta', 'var') || isempty(bConstantEta)
    bConstantEta = 0;
end
if bConstantEta~=0
    bConstantEta = 1;
end


[I,J,V] = find(P);
ne = length(I);

fnameP = tempname;
fid = fopen(fnameP, 'w+');
fwrite(fid, [nn ne], 'uint64');
fwrite(fid, I-1, 'uint64');
fwrite(fid, J-1, 'uint64');
fwrite(fid, V, 'double');
fclose(fid);

fnameWeights = tempname;
fid = fopen(fnameWeights, 'w+');
fwrite(fid, weights, 'double');
fclose(fid);

fnameY0 = tempname;
fid = fopen(fnameY0, 'w+');
fwrite(fid, Y0', 'float');
fclose(fid);

fnameY = tempname;

prog = 'sce';
cmd_str = sprintf('./%s 1 %s %s %s %s %d %f %d %d %d %f %d', ...
    prog, fnameP, fnameY, fnameWeights, fnameY0, maxIter, eta0, nRepuSamp, blockSize, blockCount, alpha, bConstantEta);

status = system(cmd_str);

fid = fopen(fnameY);
res = textscan(fid, '%f %f', 'CollectOutput', 1);
fclose(fid);
Y = res{1};

delete(fnameP);
delete(fnameY);
delete(fnameWeights);
delete(fnameY0);
