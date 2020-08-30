classdef StyleNet < handle
    
    properties(Access=private)
        Options        (1,1) Nets.Options
        NoiseAmplitude (2,1) int16 = [-20 20]
        NoiseRatio     (1,1) double = 0.7
        NoiseType      (1,1) categorical = "White"
        
        LearningRate (1,1) double = 2
    end
    properties(Access=protected)
        Logger
        ImageSize           (1,2) uint16
        Net                 (1,1) % SeriesNetwork
        Layers              (:,1) % nnet.cnn.layer.Layer
        LGraph              (1,1) % nnet.cnn.LayerGraph
        DLNet               (1,1) % dlnetwork
        
    end
    properties
        MeanVggNet
    end

    methods(Static)
        function m = Factory(Type,Logger)
            arguments
                Type (1,1) string
                Logger (1,1)
            end
            switch Type
                case "vgg19"
                    m = Nets.StyleNetVgg19(Logger,0);
                case "vgg19 + norm 1"
                    m = Nets.StyleNetVgg19(Logger,1);
                case "vgg19 + norm 2"
                    m = Nets.StyleNetVgg19(Logger,2);
                case "vgg19 + norm 3"
                    m = Nets.StyleNetVgg19(Logger,3);
                otherwise
                    error("Unrecognized Net name: " + Type)
            end
            m.Options = Nets.Options.Factory(Type);
        end
    end
    
    methods
        function m = StyleNet(Logger)
            m.Logger = Logger;
        end
        
        function Log(m,t)
            m.Logger.Log(t);
        end
        
        function SetImageSize(m,Size)
            arguments
                m
                Size (1,2) uint16 
            end
            if any(m.ImageSize~=Size)
                m.ImageSize=Size;
                m.Log("Learning rate changed to " + num2str(Size) );
            end
        end
        
        function SetLearningRate(m,Rate)
            arguments
                m
                Rate (1,1) double {mustBePositive(Rate)} = 2
            end
            if m.LearningRate~=Rate
                m.LearningRate=Rate;
                m.Log("Learning rate changed to " + Rate );
            end
        end
        
        function SetWeights(m,WeightsInGui)
            arguments
                m
                WeightsInGui (1,:) single
            end
            if m.Options.SetStyleWeights(WeightsInGui)
                m.Log("Style Weights changed to " + num2str(round(WeightsInGui*10)/10));
            end
        end
        
        function WeightChanged(m,N,V)
            m.Options.SetStyleWeight(N,V)
            m.Log("Style Weight " + N + " changed to " + num2str(round(V*10)/10));
        end
        
        function SetNoise(m,Ampl,Rate,Type)
            arguments
                m
                Ampl (1,1) int16  {mustBeNonnegative(Ampl), mustBeLessThanOrEqual(Ampl,100)}  = 20
                Rate (1,1) double {mustBeNonnegative(Rate), mustBeLessThanOrEqual(Rate,1)}    = 0.7
                Type (1,1) categorical = "White"
            end
            if m.NoiseRatio        ~= Rate ...
            || m.NoiseAmplitude(2) ~= Ampl ...
            || m.NoiseType         ~= Type
                m.NoiseAmplitude = [-Ampl Ampl];
                m.NoiseRatio     = Rate;
                m.NoiseType      = Type;
                m.Log("Noise changed to " + string(Type) ...
                        + " +/-" + Ampl + "*" + Rate );
            end
        end
        
        function PlotLayerGraph(m,What)
%           plot(m.LGraph); title("Feature Extraction Network")        
            switch What
                case "Before"
                    analyzeNetwork(m.Net)
                case "After"
                    analyzeNetwork(m.Layers)
            end
        end
        
        function Images = PreprocessImages(m,StyleImage,InputImage)
            styleImg   = imresize(StyleImage,m.ImageSize);
            contentImg = imresize(InputImage,m.ImageSize);
            
            imgInputLayer = m.LGraph.Layers(1);
            m.MeanVggNet = imgInputLayer.Mean(1,1,:);
            
            Images.style   = rescale(single(styleImg),0,255) - m.MeanVggNet;
            Images.content = rescale(single(contentImg),0,255) - m.MeanVggNet;            
        end        
        
        function TransferImage = InitializeTransferImage(m,ContentImg)
            % called by app.PrepareButtonPushed
            switch m.NoiseType
                case "White"
            RandImage = randi(m.NoiseAmplitude,[m.ImageSize 3]);
                case "Normal"
            RandImage = double(m.NoiseAmplitude(1))*randn([m.ImageSize 3]);
            end
            TransferImage = m.NoiseRatio.*RandImage + (1-m.NoiseRatio).*ContentImg;
        end        
        
        function Features = Forward(m,DLs)
            Features.Content = cell(1,numel(m.Options.ContentFeatureLayerNames));
            Features.Style   = cell(1,numel(m.Options.StyleFeatureLayerNames));
           [Features.Content{:}] = forward(m.DLNet, DLs.Content,'Outputs',m.Options.ContentFeatureLayerNames);
           [Features.Style{:}]   = forward(m.DLNet, DLs.Style,  'Outputs',m.Options.StyleFeatureLayerNames);
        end
        
        function Losses = DLFEvel(m,Runner)                
            arguments
                m
                Runner (1,1) Nets.Runner
            %<- struct Total, Content, Style
            end
            % Evaluate the transfer image gradients and state using dlfeval and the
            % local imageGradients function hereinbelow.
            [Gradients,Losses] = dlfeval(@ImageGradients ...
                                    , m.DLNet ...
                                    , Runner.DLs.Transfer ...
                                    , Runner.Features.Content ...
                                    , Runner.Features.Style ...
                                    , m.Options.ToStruct);
            
            [Runner.DLs.Transfer,Runner.TrailingAvg,Runner.TrailingAvgSq] = adamupdate( ...
                Runner.DLs.Transfer, ...
                Gradients, ...
                Runner.TrailingAvg, Runner.TrailingAvgSq, ...
                double(Runner.Iteration), ...
                m.LearningRate);
        end
    end
end

function [Gradients,Losses] = ImageGradients( mDLNet ...
                            , RunnerDLsTransfer ...
                            , RunnerFeaturesContent ...
                            , RunnerFeaturesStyle ...
                            , OptionsStruct)
    arguments    
        mDLNet
        RunnerDLsTransfer 
        RunnerFeaturesContent 
        RunnerFeaturesStyle 
        OptionsStruct
    %<- Gradients
    %<- struct Total, Content, Style
    end

    TransferFeatures = ForwardTransfer(mDLNet,RunnerDLsTransfer,OptionsStruct);
    TheContentLoss = ContentLoss(OptionsStruct.contentFeatureLayerWeights ...
                                ,TransferFeatures.Content ...
                                ,RunnerFeaturesContent);
    TheStyleLoss   = StyleLoss(  OptionsStruct.styleFeatureLayerWeights ...
                                ,TransferFeatures.Style ...
                                ,RunnerFeaturesStyle);
    TheFinalLoss = OptionsStruct.alpha*TheContentLoss + OptionsStruct.beta*TheStyleLoss;
    Gradients = dlgradient(TheFinalLoss,RunnerDLsTransfer);
    Losses = struct( "Total",   gather(extractdata(TheFinalLoss))   , ...
                     "Content", gather(extractdata(TheContentLoss)) , ...
                     "Style",   gather(extractdata(TheStyleLoss)) );

end

function Loss = ContentLoss(mOptionsContentFeatureLayerWeights,TransferContentFeatures,ContentFeatures)
    Loss = 0;
    for i=1:numel(ContentFeatures)
        Loss = Loss + ...
            0.5 * mOptionsContentFeatureLayerWeights(i) ...
                * mean((TransferContentFeatures{1,i}-ContentFeatures{1,i}).^2,'all');
    end
end

function Loss = StyleLoss(mOptionsStyleFeatureLayerWeights,TransferStyleFeatures,StyleFeatures)
    Loss = 0;
    for i=1:numel(StyleFeatures)
        SF = StyleFeatures{1,i};
        Loss = Loss + ...
            mOptionsStyleFeatureLayerWeights(i) ...
          * mean((ComputeGramMatrix(TransferStyleFeatures{1,i})-ComputeGramMatrix(SF)).^2,'all')/(numel(SF)^2);
    end
end


function TransferFeatures = ForwardTransfer(mDLNet,DLsTransfer,OptionsStruct)
    TransferFeatures.Content = cell(1,numel(OptionsStruct.contentFeatureLayerNames));
    TransferFeatures.Style   = cell(1,numel(OptionsStruct.styleFeatureLayerNames));
   [TransferFeatures.Content{:}] = forward(mDLNet, DLsTransfer, 'Outputs',OptionsStruct.contentFeatureLayerNames);
   [TransferFeatures.Style{:}]   = forward(mDLNet, DLsTransfer, 'Outputs',OptionsStruct.styleFeatureLayerNames);
end


function GramMatrix = ComputeGramMatrix(FeatureMap)
    [H,W,C] = size(FeatureMap);
    ReshapedFeatures = reshape(FeatureMap,H*W,C);
    GramMatrix = ReshapedFeatures' * ReshapedFeatures;
end

