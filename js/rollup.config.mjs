import nodePolyfills from 'rollup-plugin-polyfill-node';
import nodeResolve from '@rollup/plugin-node-resolve';
import typescript from 'rollup-plugin-typescript2';
import commonjs from '@rollup/plugin-commonjs';
import terser from '@rollup/plugin-terser';
import dts from "rollup-plugin-dts";

export default [
    {
        input: [
            'src/index.ts',
        ],
        output: {
            file: 'dist/tvm-rt-wasm.js',
            format: 'cjs',
            sourcemap: false,
        },
        plugins: [commonjs(), typescript(), nodeResolve(), nodePolyfills(), terser()]
    },
    {
        input: [
            'dist/src/index.d.ts',
        ],
        output: {
            file: 'dist/tvm-rt-wasm.d.ts',
            format: 'es',
            sourcemap: false,
        },
        plugins: [dts()]
    }
];
