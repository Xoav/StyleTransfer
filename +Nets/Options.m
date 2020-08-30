classdef Options < handle
    properties
        ContentFeatureLayerNames   (1,1) string 
        ContentFeatureLayerWeights (1,1) double 
        StyleFeatureLayerNames     (1,:) string 
        StyleFeatureLayerWeights   (1,:) double 
        Alpha (1,1) double % Total Loss
        Beta  (1,1) double % Total Loss
    end
    methods
        function S = ToStruct(m)
            S = struct( ...
            "contentFeatureLayerNames",   m.ContentFeatureLayerNames, ...
            "contentFeatureLayerWeights", m.ContentFeatureLayerWeights, ...
            "styleFeatureLayerNames",     m.StyleFeatureLayerNames, ...
            "styleFeatureLayerWeights",   m.StyleFeatureLayerWeights, ...
            "alpha",m.Alpha, "beta", m.Beta );
        end
        
         function Changed = SetStyleWeights(m,StyleWeights)
            arguments
                m
                StyleWeights (1,5) single
            end
            Changed = ~all(m.StyleFeatureLayerWeights==StyleWeights);
            if Changed
                m.StyleFeatureLayerWeights = StyleWeights;
            end
         end
         
         function SetStyleWeight(m,N,StyleWeight)
            arguments
                m
                N (1,1) uint8
                StyleWeight (1,1) single
            end
            m.StyleFeatureLayerWeights(N) = StyleWeight;
         end
    end
    methods(Static)
        function m = Factory(Type)
            arguments
                Type (1,1) string
            end
        m = Nets.Options;
            switch Type
                case {"vgg19" ...
                      "vgg19 + norm 1" ...
                      "vgg19 + norm 2" ...  
                      "vgg19 + norm 3"}  
        m.ContentFeatureLayerNames    = "conv5_4"; 
        m.ContentFeatureLayerWeights  = 1;
        m.StyleFeatureLayerNames      = ["conv1_1" "conv2_1" "conv3_1" "conv4_1" "conv5_1"];
        m.StyleFeatureLayerWeights    = [ 1         1         1         1         1       ];    
        m.Alpha=1;   m.Beta = 1; 
                case "vgg19 Fred's original"
        m.ContentFeatureLayerNames    = "conv4_2"; 
        m.ContentFeatureLayerWeights  = 1;
        m.StyleFeatureLayerNames      = ["conv1_1" "conv2_1" "conv3_1" "conv4_1" "conv5_1"];
        m.StyleFeatureLayerWeights    = [ 0.5       1.0       1.5       3.0       4.0     ];    
        m.Alpha=1;   m.Beta = 1E3;  
                otherwise
                    error("Unrecognized Net name: " + Type)                   
            end
        end
    end
end

