import antfu from '@antfu/eslint-config';
import jestDom from 'eslint-plugin-jest-dom';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import tailwind from 'eslint-plugin-tailwindcss';

export default antfu({
  react: true,
  typescript: true,

  lessOpinionated: true,
  isInEditor: false,

  stylistic: {
    semi: true,
  },

  formatters: {
    css: true,
  },

  ignores: [
    './node_modules/**/*',
    '.idea/**/*',
  ],
}, ...tailwind.configs['flat/recommended'], jsxA11y.flatConfigs.recommended, {

  rules: {

  },
}, {
  files: ['**/*.test.ts?(x)'],
  ...jestDom.configs['flat/recommended'],
}, {
  files: ['**/*.spec.ts', '**/*.e2e.ts'],
}, {
  rules: {
    '@typescript-eslint/no-explicit-any': 'error',
    'ts/ban-ts-comment': 'off',
    'jsonc/sort-keys': 'off',
    'no-useless-catch': 'off',
    'antfu/no-top-level-await': 'off', // Allow top-level await
    'style/brace-style': ['error', '1tbs'], // Use the default brace style
    'ts/consistent-type-definitions': ['error', 'type'], // Use `type` instead of `interface`
    'react/prefer-destructuring-assignment': 'off', // Vscode doesn't support automatically destructuring, it's a pain to add a new variable
    'node/prefer-global/process': 'off', // Allow using `process.env`
    'test/padding-around-all': 'error', // Add padding in test files
    'test/prefer-lowercase-title': 'off', // Allow using uppercase titles in test titles,
    'import/no-import-module-exports': [
      'error',
      {
        exceptions: ['**/*/webpack/**/*'],
      },
    ],
    'arrow-body-style': 'off',
    'style/arrow-parens': 'off',
    'arrow-parens': 'error',
    'consistent-return': 'error',
    'default-param-last': 'error',
    'no-nested-ternary': 'off',
    'react/no-context-provider': 'off',
    'import/no-extraneous-dependencies': ['error', { devDependencies: true }],
    'no-console': [
      'error',
      {
        allow: ['info', 'error', 'warn'],
      },
    ],
    // 'padding-line-between-statements': [
    //   'error',
    //   { blankLine: 'any', prev: '*', next: 'return' },
    //   { blankLine: 'any', prev: ['block-like'], next: '*' },
    //   { blankLine: 'any', prev: ['const', 'let', 'var'], next: '*' },
    //   {
    //     blankLine: 'any',
    //     prev: ['const', 'let', 'var'],
    //     next: ['const', 'let', 'var'],
    //   },
    // ],
    'no-use-before-define': 'error',
    'no-restricted-syntax': 'error',
    'no-multi-assign': 'error',
    'no-promise-executor-return': 'error',
    'no-underscore-dangle': 'off',
  },
});
