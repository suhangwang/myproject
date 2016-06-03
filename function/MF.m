function[U, V, performance] = MF(R, K, W, testA, alpha, stepSize, maxiter)
%Input-
%R  user-item rating matrix
%S  user-user social relations
%W  weight of the matrix
%lambda controlling the contribution from local trust
%k  # of latent factors
%X staic-user matrix

%Output-
%U user preference
%V item characteristics
%h regression coefficients
%b user specific bias



% initilize the variables
[nuser,nitem] = size(R);
U = rand(K,nuser);
V = rand(K, nitem);

iter = 0;
% gamma = 0.00005;
gamma = stepSize;

count = 1;
test_pos_ind =find(testA>0);
test_neg_ind = find(testA<0);
Y = [ones(length(test_pos_ind),1); -1*ones(length(test_neg_ind),1)];
while(iter < maxiter)

    iter = iter + 1

    %Constuct W .* W
    %W = idxR;

    %Update U, V, and H
    UV = U'*V;

    gradU = 2*( - V*(W.*R)' + V*(W.*UV)' + alpha*U);
    gradV = 2*( - U*(W.*R) + U*(W.*UV) + alpha*V);

    U = U - gamma*gradU;
    V = V - gamma*gradV;

    if iter >= 10 && mod (iter,10) == 0
        Rhat = U'*V;
        predY = [Rhat(test_pos_ind); Rhat(test_neg_ind)];
        predY_binary = predY;
        predY_binary(predY_binary>=0) = 1;
        predY_binary(predY_binary<0) = -1;
        AUC1 = fastAUC(Y, predY, 1);
        AUC2 = fastAUC(Y, predY_binary, 1);
    end

end

cdend