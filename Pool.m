function Y2 = Pool(Y1)
    Y2 = (Y1(1:2:end,1:2:end,:)+Y1(2:2:end,1:2:end,:)+Y1(1:2:end,2:2:end,:)+Y1(2:2:end,2:2:end,:))/4;
end