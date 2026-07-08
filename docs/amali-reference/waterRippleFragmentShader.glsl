#version 300 es
        precision highp float;

        in vec2 vTextureCoord;
        in vec3 vVertexPosition;

        uniform sampler2D uTexture;
        uniform sampler2D uBgTexture;
        uniform sampler2D uPingPongTexture;
        uniform vec2 uResolution;

        uniform float uStrength;
        uniform float uRadius;
        uniform float uScale;
        uniform float uTime;
        uniform float uDispersion;
        uniform float uViscosity;
        uniform float uLighting;
        uniform int uEffectType;
        uniform int uPass;
        uniform vec3 uColor;

        const float PI = 3.1415926;
        const float ITERATIONS = 24.0;

        float getGaussianWeight(int i) {
            if (i == 0) return 0.20236;
            if (i == 1) return 0.19618;
            if (i == 2) return 0.17825;
            if (i == 3) return 0.15175;
            if (i == 4) return 0.12099;
            if (i == 5) return 0.09029;
            if (i == 6) return 0.06272;
            if (i == 7) return 0.04096;
            if (i == 8) return 0.02520;
            if (i == 9) return 0.01464;
            if (i == 10) return 0.00804;
            if (i == 11) return 0.00417;
            return 0.0;
        }

        out vec4 fragColor;

        mat2 rot(float a) {
            float c = cos(a);
            float s = sin(a);
            return mat2(c, -s, s, c);
        }

        float luma(vec3 color) {
            return dot(color, vec3(0.299, 0.587, 0.114));
        }

        vec3 chromatic_aberration(vec3 color, vec2 uv) {
        vec2 offset = (uv - vTextureCoord) * (uDispersion * 0.2);
        vec4 left = texture(uBgTexture, uv - offset);
        vec4 right = texture(uBgTexture, uv + offset);

        color.r = left.r;
        color.b = right.b;

        return color;
        }

        vec4 blur(vec2 uv, vec2 dir) {
        vec4 color = vec4(0.0);
        float total_weight = 0.0;
        
        vec4 center = texture(uTexture, uv);
        float center_weight = getGaussianWeight(0);
        color += center * center_weight;
        total_weight += center_weight;
        
        for (int i = 1; i <= 11; i++) {
            float weight = getGaussianWeight(i);
            float offset = mix(0.005, 0.015, uViscosity) * float(i)/11.;
            
            vec4 sample1 = texture(uTexture, uv + offset * dir);
            vec4 sample2 = texture(uTexture, uv - offset * dir);
            
            color += (sample1 + sample2) * weight;
            total_weight += 2.0 * weight;
        }

        return color / total_weight;
        }

        vec3 calculateNormal(sampler2D tex, vec2 uv) {
            float stengthScale = mix(3., 7., uScale);
            float stepScale = mix(1., 3., uScale);
            float strength = mix(1., stengthScale, uStrength);
            float stepSize = mix(1., stepScale, uStrength);
            float step = stepSize / 1080.;
            
            float left = texture(tex, uv + vec2(-step, 0.0)).r;
            float right = texture(tex, uv + vec2(step, 0.0)).r;
            float top = texture(tex, uv + vec2(0.0, -step)).r;
            float bottom = texture(tex, uv + vec2(0.0, step)).r;
            
            vec3 normal;
            normal.x = (right - left) * strength;
            normal.y = -(bottom - top) * strength;
            normal.z = -1.0;
            
            return normalize(normal);
        }

        vec2 calculateRefraction(vec3 normal, float ior) {
            vec3 I = vec3(0.0, 0.0, 1.0);
        
            float ratio = 1.0 / ior;
            vec3 refracted = refract(I, normal, ratio);
            float refractionScale = mix(0.2, 0.4, uScale);
            
            float refractionAmount = mix(0.01, refractionScale, uStrength);
            return refracted.xy * refractionAmount;
        }

        vec4 drawRipple(vec2 uv) {
            vec2 scaled = mix(uv, (uv - 0.5) * 0.5 + 0.5, uScale);
            vec3 normal = calculateNormal(uPingPongTexture, scaled);
            return vec4(normal, 1.);
        }

        const vec3 LIGHT_POS = vec3(2.0, 2.0, 3.0);
        const vec3 VIEW_POS = vec3(0.0, 0.0, 2.0); 
        const float SPECULAR = 2.4;
        const float SHININESS = 128.0;

        vec3 calculateLighting(vec3 normal, vec2 uv) {
        vec3 N = normal;
        vec3 worldPos = vec3(uv * 2.0 - 1.0, 0.0);
        vec3 lightDir = normalize(LIGHT_POS - worldPos);
        vec3 viewDir = normalize(VIEW_POS - worldPos);
        vec3 reflectDir = reflect(-lightDir, N);
        
        float diff = max(dot(N, lightDir), 0.0);
        vec3 diffuse = vec3(diff);
        
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), SHININESS);
        vec3 specular = vec3(spec * SPECULAR);
        
        return diffuse + specular;
        }

        vec4 getRipple(vec2 uv) {  
        vec3 normal = texture(uTexture, uv).rgb;
        
        vec2 refractionOffset = calculateRefraction(normal, 1.333);
        vec2 refractedUv = uv + refractionOffset;
        
        vec3 refractedNormal = texture(uTexture, refractedUv).rgb;

        vec4 refractedColor = texture(uBgTexture, refractedUv);
        refractedColor.rgb = chromatic_aberration(refractedColor.rgb, refractedUv);

        vec3 caustics = calculateLighting(refractedNormal, refractedUv);
        float causticsShadow = dot(normal, normalize(vec3(2.0, -2.0, 3.0) - vec3(uv * 2.0 - 1.0, 0.0))) + 1.;
        
        float shadowFactor = causticsShadow;
        vec3 lightingFactor = caustics;

        shadowFactor = mix(1., shadowFactor, uLighting);
        lightingFactor = mix(vec3(0), lightingFactor * uColor, uLighting);
        
        vec4 finalColor = vec4(refractedColor.rgb - vec3(1.-shadowFactor) * uColor + lightingFactor, refractedColor.a);
        return finalColor;
        }

        vec4 getColor(vec2 uv) {
        switch(uPass) {
            case 0: return drawRipple(uv); break;
            case 1: return blur(uv, vec2(1, 0)); break;
            case 2: return blur(uv, vec2(0, 1)); break;
            case 3: return getRipple(uv); break;
            default: return drawRipple(uv);
        }
        }

        void main() {
            vec2 uv = vTextureCoord;
            vec4 color = getColor(uv);
            fragColor = color;
        }
    