classdef StyleNetVgg19 < Nets.StyleNet

    properties(Access=private)
        LastFeatureLayerIdx (1,1) uint16 = 38 % the last pooling layer
        MaxImageSizeOnThisGPU = [ 900 900 ]
    end
    
    methods
        function m = StyleNetVgg19(Logger,BatchNorm)
            m@Nets.StyleNet(Logger)
            m.Net = vgg19;
            m.Layers = m.Net.Layers;
            m.Layers = m.Layers(1:m.LastFeatureLayerIdx);  
            m.ReplaceMaxByAveragePoolingLayers;
            m.AddBatchNormalizationLayers(BatchNorm);
            m.LGraph = layerGraph(m.Layers);
            m.DLNet = dlnetwork(m.LGraph);
            m.Log("vgg19 net constructed.");
            m.ImageSize = m.MaxImageSizeOnThisGPU;
        end
        
    end
    
    methods(Access=protected)
        
        function AddBatchNormalizationLayers(m,Option)
            After = "nnet.cnn.layer.AveragePooling2DLayer";
            switch Option
                case 1
                    At = [ 0 0 1 0 ];
                case 2
                    At = [ 0 1 0 0 ];
                case 3
                    At = [ 0 1 1 0 ];
                otherwise
                    m.Log("Using no batch normalization");
                    return
            end
            N=0;
            for L = (numel(m.Layers)-1):-1:1
                Layer = m.Layers(L);
                if isa(Layer,After)
                    N=N+1;
                    if At(N)
                        m.Layers = [m.Layers(1:L);
                                    batchNormalizationLayer("Name","Norm"+N);
                                    m.Layers((L+1):end)];
                        m.Log("Inserted Batch Normalization after " + L + " " + After);
                    end
                end
            end
            
        end
        
        function ReplaceMaxByAveragePoolingLayers(m)
%       Layers that will be found to be Max Pooling = [ 6 11 20 29 38 ]; 
            for L = 1:numel(m.Layers)
                Layer = m.Layers(L);
                if isa(Layer,'nnet.cnn.layer.MaxPooling2DLayer')
                    m.Layers(L) = averagePooling2dLayer(Layer.PoolSize,'Stride',Layer.Stride,'Name',Layer.Name);
                end
            end
        end
    end
end

