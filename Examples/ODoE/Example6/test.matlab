n = 7;
width = 1/(n-1);

% generate sample (n x n x n factorial design for 3 parameters)
% X - [0,1], Y - [0, 2], Z - [0, 4]
X = zeros(n*n*n, 3);
m = 0;
for ii = 1 : n
  xval = (ii - 1) * width;
  for jj = 1 : n
    yval = (jj - 1) * width * 2;
    for kk = 1 : n
      zval = (kk - 1) * width * 4;
      m = m + 1;
      X(m,1) = xval;
      X(m,2) = yval;
      X(m,3) = zval;
    end
  end
end

% create information matrix and its inverse
% prepare X1, which has one more row to accommodate design
A = X' * X;
invA = inv(A);
N = n * n * n;
X1 = zeros(N+1,3);
X1(1:N,:) = X;

% generate test (100 points)
M = 100;

% iterate to test repeatedly
while 1

  T = rand(M,3);

% D-optimal
% maxdet = 0;
% indexd = -1;
% for ii = 1 : M
%   X1(N+1,:) = T(ii,:);
%   A1 = X1' * X1;
%   ddet = det(A1);
%   if (ddet > maxdet)
%     maxdet = ddet;
%     indexd = ii;
%   end
% end;
% disp(['Max determinant at Test = ' int2str(indexd) ' maxdet = ' num2str(maxdet)])

WA = zeros(M,1);
% W-optimal (max variance)
  maxw = -999;
  indexw = -1;
  for ii = 1 : M
    dtmp = T(ii,:) * invA * T(ii,:)';
    WA(ii) = dtmp;
    if (dtmp > maxw)
      maxw = dtmp;
      indexw = ii;
    end
    %disp(['W at ' int2str(ii) ' = ' num2str(dtmp)])
  end;
  disp(['W-optimal at Test = ' int2str(indexw) ' maxW = ' num2str(maxw)])

WG = zeros(M,1);
  minG = 1e46;
  indexg = -1;
  hasProblem = 0;
  for ii = 1 : M
     X1(N+1,:) = T(ii,:);
     A1 = X1' * X1;
     invA1 = inv(A1);
     % for this particular design, find candidate that gives largest variance
     maxG = -999;
     indexG = -1;
     for jj = 1 : M
       dtmp = T(jj,:) * invA1 * T(jj,:)';
       %isp(['G- at Test = ' int2str(jj) ' dtmp = ' num2str(dtmp)])
       if (dtmp > maxG)
         maxG = dtmp;
         indexG = jj;
       end;
     end;
     disp(['Max G- for Candidate = ' int2str(ii) ' maxG = ' num2str(maxG) ' indexG = ' int2str(indexG)])
WG(ii) = maxG;
     % if maximum candidate is not the design itself, analyze
     if (indexG ~= ii)
       %disp(['Max ERROR Design=' int2str(ii) ' MaxCand=' int2str(indexG)])
       a = T(indexG,:) * invA * T(indexG,:)';
       b = T(ii,:) * invA * T(indexG,:)';
       c = T(ii,:) * invA * T(ii,:)';
       %disp(['Using MaxCand: ' num2str(a) ' ' num2str(b) ' ' num2str(c) ' ' num2str(a-b*b/(1+c))])
       c = T(ii,:) * invA * T(ii,:)';
       b = T(ii,:) * invA * T(ii,:)';
       a = T(ii,:) * invA * T(ii,:)';
       %disp(['Using Design: ' num2str(a) ' ' num2str(b) ' ' num2str(c) ' ' num2str(a-b*b/(1+c))])
       %disp(['Not equal(GInd,CandI): ' num2str(indexG) ' vs ' num2str(ii)])
       %pause
     end
     if (maxG < minG)
       minG = maxG;
       %disp(['indexg switched from ' int2str(indexg) ' to ' int2str(ii)])
       indexg = ii;
     end;
     if (ii == indexw & indexG ~= ii)
       disp(['Problem Design=' int2str(ii) ' MaxCand=' int2str(indexG)])
       disp('Problem')
       a = T(indexG,:) * invA * T(indexG,:)';
       b = T(ii,:) * invA * T(indexG,:)';
       c = T(ii,:) * invA * T(ii,:)';
       disp(['Using MaxCand: ' num2str(a) ' ' num2str(b) ' ' num2str(c) ' ' num2str(a-b*b/(1+c))])
       c = T(ii,:) * invA * T(ii,:)';
       b = T(ii,:) * invA * T(ii,:)';
       a = T(ii,:) * invA * T(ii,:)';
       disp(['Using Design: ' num2str(a) ' ' num2str(b) ' ' num2str(c) ' ' num2str(a-b*b/(1+c))])
       hasProblem = 1;
       pause
     end
     %disp('Press Enter')
  end;
  disp(['W-optimal at Candidate = ' int2str(indexw) ' maxW = ' num2str(maxw)])
  disp(['G-optimal at Candidate = ' int2str(indexg) ' minG = ' num2str(minG)])
  %dtemp = T(indexg,:) * invA * T(indexg,:)'
  %dtemp = T(indexw,:) * invA * T(indexw,:)'

  clf
  maxa = max(WA);
  mina = min(WA);
  WA = (WA-mina)/(maxa-mina);
  plot(WA,'linewidth',2)
  grid minor
  box on
  grid on
  set(gca,'linewidth',2)
  set(gca,'fontsize',12)
  hold on
  maxa = max(WG);
  WG = maxa - WG;
  maxa = max(WG);
  mina = min(WG);
  WG = (WG-mina)/(maxa-mina);
  plot(WG,'linewidth',2)
  legend('W-metric', 'G-metric')
  if indexw ~= indexg
    disp(['Mismatch: W = ' int2str(indexw) ' , G = ' int2str(indexg)])
    %pause
  end;
  if hasProblem == 1
    disp(['W-index = ' int2str(indexw) ' , G-index ' int2str(indexg)])
    pause
  end 
  %pause
end

maxw = -999;
indexw = -1;
for ii = 1 : M
  dtmp = T(ii,:) * invA * T(ii,:)';
  if (dtmp >= maxw)
    maxw = dtmp;
    indexw = ii;
  end
  [ii dtmp]
end;
disp(['W-optimal at Test = ' int2str(indexw) ' maxW = ' num2str(maxw)])
 


