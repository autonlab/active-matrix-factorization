function lbfstatus(t,f,x,auxdata)

persistent myoutputprevstring;
persistent lastdisplay;
persistent timer;
persistent fvec
persistent savename;
persistent lbfstatuswindow1;
persistent lbfstatuswindow2;

global GLOBALITERS;

if (nargin==0)
    fvec = [];
    savename = ['lbfgs' datestr(now,'mm-dd-yy_HH:MM:SS') '.mat'];
    %figure(101);
    
    %subplot(1,3,[1 2]);
    %lbfstatuswindow1 = gca;
    %subplot(1,3,3);
    %lbfstatuswindow2 = gca;
    return;
end

if (t==1)
    myoutputprevstring = '';
    timer = tic;
    lastdisplay = toc(timer);
    %    fvec = f;
end

fvec = [fvec; f];
GLOBALITERS = length(fvec);
if (toc(timer)>lastdisplay+.5)
    fprintf(repmat('\b',1,length(myoutputprevstring)));
    
    myoutputprevstring = ...
        sprintf('iter=%d, f(x)=%f, time=%f seconds',t,f,toc(timer));
    fprintf('%s',myoutputprevstring);

    %subplot(1,3,[1 2]);
    %plot(lbfstatuswindow1, fvec,'x');
    %xlabel(lbfstatuswindow1, 'Iteration');
    %ylabel(lbfstatuswindow1,'Objective Value');
    %subplot(1,3,3);
    iterx = max(1,length(fvec)-10):length(fvec);
    %plot(lbfstatuswindow2, iterx, fvec(iterx),'x');
    %xlabel(lbfstatuswindow2, 'Iteration');
    %ylabel(lbfstatuswindow2,'Objective Value');
    %drawnow;
    
    %lastdisplay = toc(timer);

    %save(savename, 'fvec');
end
