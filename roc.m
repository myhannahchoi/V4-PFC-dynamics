%%% Usage roc(category1, category2, rocid)
%%% rocid = 1 means category1 is pref category; category2 is pref otherwise
function [rocarea] = roc(category1, category2, rocid)
if(rocid == 1)
    cat2 = category1;
    cat1 = category2;
else
    cat1 = category1;
    cat2 = category2;
end
rmin = min([cat1 cat2])-1;
rcrit = fliplr(sort([cat1 cat2 rmin]));
rocfa = sum(gt(repmat(cat1', 1, length(rcrit)), repmat(rcrit, length(cat1), 1)), 1);
rochits = sum(gt(repmat(cat2', 1, length(rcrit)), repmat(rcrit, length(cat2), 1)), 1);
rocfa = rocfa./length(cat1);
rochits = rochits./length(cat2);
rocarea = sum(diff(rocfa).*(rochits(1:end-1)+rochits(2:end))./2.0);

%%% rocid represents the preferred category.
%rocarea = 0.0;
%for(i=1:length(cat2))
%     for(j=1:length(cat1))
%         if(cat2(i) > cat1(j))
%             rocarea = rocarea + 2.0;
%         elseif (cat2(i) == cat1(j))
%             rocarea = rocarea + 1.0;
%         end;
%     end;
% end;
% rocarea = rocarea./(length(cat1)*length(cat2)*2.0);