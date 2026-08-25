/**
 * Stand-in for `electron` when bundling main-process modules for the test suite.
 *
 * Modules under src/main import Electron at the top level even when the logic under test is
 * pure — storage/db.ts reaches it only through storage/paths.ts, which already honours
 * LLMM_APPDATA_DIR and so never asks Electron for a directory during a test run. Aliasing the
 * import to this file is what lets those modules load under plain Node at all.
 *
 * Deliberately minimal: anything a test actually depends on should be overridden through the
 * real environment variables, not faked here.
 */

const notAvailable = (name) => () => {
  throw new Error(`electron.${name} is not available in the test bundle — set the matching env override instead`)
}

export const app = {
  isPackaged: false,
  getPath: notAvailable('app.getPath'),
  getAppPath: notAvailable('app.getAppPath')
}

export const safeStorage = {
  isEncryptionAvailable: () => false,
  encryptString: notAvailable('safeStorage.encryptString'),
  decryptString: notAvailable('safeStorage.decryptString')
}

export default { app, safeStorage }
