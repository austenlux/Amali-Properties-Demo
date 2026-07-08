
        precision mediump float;

        attribute vec3 aVertexPosition;
        attribute vec2 aTextureCoord;

        uniform mat4 uMVMatrix;
        uniform mat4 uPMatrix;
        uniform vec2 uUvScale;
        uniform vec2 uUvOffset;

        varying vec2 vTextureCoord;
        varying vec3 vVertexPosition;

        void main() {
            gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
            // Apply object-fit: cover transform to UV coordinates
            vTextureCoord = aTextureCoord * uUvScale + uUvOffset;
        }
    