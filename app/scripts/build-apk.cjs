const {spawnSync} = require('node:child_process');
const {copyFileSync, existsSync, mkdirSync, statSync} = require('node:fs');
const path = require('node:path');

const architectures = {
  'arm64-v8a': 'arm64-v8a',
  'armeabi-v7a': 'armeabi-v7a',
  x86_64: 'x86_64',
  universal: 'armeabi-v7a,arm64-v8a,x86,x86_64',
};
const target = process.argv[2] || 'arm64-v8a';
if (!Object.hasOwn(architectures, target)) {
  console.error(`Unknown APK target: ${target}`);
  process.exit(1);
}

const androidDir = path.resolve(__dirname, '../android');
const env = {...process.env};
if (process.platform === 'win32' && !env.JAVA_HOME) {
  const studioJdk = path.join(env.ProgramFiles || 'C:\\Program Files', 'Android/Android Studio/jbr');
  if (existsSync(studioJdk)) env.JAVA_HOME = studioJdk;
}
const args = [
  ':app:assembleRelease',
  `-PreactNativeArchitectures=${architectures[target]}`,
  '-PcompressNativeLibs=true',
  '--console=plain',
];
console.log(`Building standalone test APK for ${target} (compressed native libraries).`);
const build = process.platform === 'win32'
  ? spawnSync(env.ComSpec || 'cmd.exe', ['/d', '/c', 'gradlew.bat', ...args], {cwd: androidDir, env, stdio: 'inherit'})
  : spawnSync('./gradlew', args, {cwd: androidDir, env, stdio: 'inherit'});
if (build.error) console.error(build.error.message);
if (build.status !== 0) process.exit(build.status || 1);

// Preserve distinct artifacts when switching between phone and emulator builds.
const outputDir = path.join(androidDir, 'app/build/outputs/apk-test');
mkdirSync(outputDir, {recursive: true});
const output = path.join(outputDir, `water-meter-ocr-${target}.apk`);
copyFileSync(path.join(androidDir, 'app/build/outputs/apk/release/app-release.apk'), output);
console.log(`\nAPK: ${output}\nSize: ${(statSync(output).size / 1024 / 1024).toFixed(1)} MiB`);
