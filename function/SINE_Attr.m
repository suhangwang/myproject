function SINE_Attr(A, X, L, K, Pr, alpha, beta, gamma, lambda, lr, maxIter)
% initialization
[n,m] = size(X);
U = random(n,K);
P = random(n,K);
Q = random(n,K);
H = random(K,K);
V = random(m,K);

N = size(Pr,1);
M = cell(N,1);
for ind = 1:N
    M{ind} = sparse(n,n);
    i = Pr(ind,1);
    j = Pr(ind,2);
    k = Pr(ind,3);
    M{ind}(i,j) = -1;
    M{ind}(j,i) = -1;
    M{ind}(k,k) = -1;
    M{ind}(i,k) = 1;
    M{ind}(k,i) = 1;
    M{ind}(j,j) = 1;
end

W = sparse(n,n);
W(A>0) = 1;
W(A<0) = 1;

W = W .* W; % use this if W contains elements other 0 and 1

WA = W .* A;

% update parameters
iter = 0;

tmp_sum = sparse(n,n);
PPt = P*P';
for ind = 1:N
    i = Pr(ind,1);
    j = Pr(ind,2);
    k = Pr(ind,3);
    tmp = PPt(j,j) - PPt(k,k) + 2*PPt(i,k) - 2*PPt(i,j);
    if tmp > 0
        tmp_sum = tmp_sum + M{ind};
    end
end


while iter < maxIter
    % update U
    B = U + P;
    C = X - Q*V';
    BH = B * H;
    BHt = B * H';
    BHBt = BH * B';
    WBHBt = W.*BHBt;
    term1 = WA*BHt;
    term2 = WA'*BH;
    term3 = WBHBt*BHt;
    term4 = WBHBt'*BH;
    term5 = -1*term1 - term2 + term3 + term4;
    gradU = term5 + beta*U*V'*V - beta*C*V  + lambda*U;
    
    U = U - lr*gradU;
    
    % update P
    gradP = term5 + alpha*tmp_sum*P + lambda*P;
    
    P = P - lr*gradP;
    
    % update H
    B = U + P;
    
    gradH = B'*(W.*(B*H*B'))*B - B'*WA*B + lambda*H;
    
    H = H - lr*gradH;
    
    % update Q
    VtV = V'*V;
    gradQ = beta*Q*VtV - beta*X*V + beta*U*VtV + gamma*L*Q + lambda*Q;
    
    Q = Q - lr*gradQ;
    
    % update V
    term6 = U + Q;
    gradV = beta*V*term6'*term6 - beta*X'*term6 + lambda*V;
    V = V - lr*gradV;
    
    % calculate objective function value
    oterm1 = W.*(A - (U+P)*H*(U+P)');
    oterm2 = sum(sum(oterm1.*oterm1));
    oterm2 = X - (U+Q)*V';
    oterm2 = sum(sum(oterm2.*oterm2));
    
    tmp_sum = sparse(n,n);
    PPt = P*P';
    for ind = 1:N
        i = Pr(ind,1);
        j = Pr(ind,2);
        k = Pr(ind,3);
        tmp = PPt(j,j) - PPt(k,k) + 2*PPt(i,k) - 2*PPt(i,j);
        if tmp > 0
            tmp_sum = tmp_sum + M{ind};
        end
    end
    
    obj = oterm1 + alpha*trace(M*PPt) + beta*oterm2 + gamma*trace(Q'*L*Q) + lambda*(sum(sum(U.*U)) + sum(sum(P.*P)) + sum(sum(Q.*Q)) + sum(sum(V.*V)) + sum(sum(H.*H)));
    
    iter = iter + 1;
end