classdef Runner < handle
    properties(Access=private)
        Logger
        MinimumLoss (1,1) double = inf
    end
    properties % StyleNet
        DLs (1,1) % Content, Style, Transfer
        Features (1,1)  % Content, Style
        TrailingAvg = []
        TrailingAvgSq = []
        Iteration
    end
    
    
    methods
        function m = Runner(Logger)
            m.Logger = Logger;
        end

        function Log(m,t)
            m.Logger.Log(t);
        end
       
        function TrainNetwork(m,Images,NetObj)
            m.DLs = struct( ...
                "Content", gpuArray(dlarray(Images.content, 'SSC')), ...
                "Style",   gpuArray(dlarray(Images.style,   'SSC')), ...
                "Transfer",gpuArray(dlarray(Images.transfer,'SSC'))  );
            m.Features = NetObj.Forward(m.DLs); % : Content, Style
            m.MinimumLoss = inf;
            m.Log("Runner trained the network.")
        end
        
        function IterateFrom(m,NetObj,From)
            arguments
                m
                NetObj
                From (1,1) uint16
            end
            m.Iteration = From;
            while m.Logger.IsRunSwitchOn
                Losses = NetObj.DLFEvel(m);               
                if Losses.Total < m.MinimumLoss
                    m.MinimumLoss = Losses.Total;
                %   m.DLs.Output = m.DLs.Transfer;
                end
                m.DispIteration(NetObj.MeanVggNet);
                
                m.Iteration = m.Iteration + 1;
                m.Logger.UpdateIteration(m.Iteration);
            end
            m.Logger.Log("Exit iteration loop at " + m.Iteration + " since Run Switch is Off.")
        end
    end
    
    methods (Access=private)
        function DispIteration(m,ImageMean)
        % Display the transfer image on the first iteration and after every
        % some iterations. The postprocessing steps are described in the "Postprocess
        % Transfer Image for Display" section of this example.
            if mod(m.Iteration,10) ~= 0 && (m.Iteration > 1)
                return
            end
            m.Logger.OutputImage = imresize(uint8(gather(extractdata(m.DLs.Transfer))+ImageMean) ...
                        , m.Logger.OutputImageSize);            
            m.Logger.ShowOutput("Iteration " + m.Iteration);
%           m.Logger.Log("At iteration " + m.Iteration);
        end
    end
end

