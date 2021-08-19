fprintf('Before running the demo, make sure you have downloaded the pre-computed similarity matrix in the current folder.\n');

load P_SHUTTLE.mat
Y = sce(P, 1e5);
DisplayVisualization(Y);