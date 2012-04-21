load demo_data;
%loads Xtr and Xte training and testing matrices

%% subsample. Comment this out if you have time to wait for the full job

rows = rand(size(Xtr,1),1) < .3;
cols = rand(size(Xtr,2),1) < .3;

Xtr = Xtr(rows, cols);
Xte = Xte(rows, cols);


%% mask is a binary sparse matrix of the query entries
mask = Xte>0;

delta = 1.5;

[E, P, vals, lagrange] = ...
    ratingconcentration(Xtr, mask, @sets_square5, delta);

%% pick some random ratings and show prediction

[I,J,V] = find(Xte);
figure(1);
for i = 1:4
    subplot(4,1,i);
    ind = randi(nnz(mask));
    
    bar(P(ind,:));
    xlabel('Rating');
    ylabel('Probability');
    
    title(sprintf('True rating: %d', V(ind)));
end

