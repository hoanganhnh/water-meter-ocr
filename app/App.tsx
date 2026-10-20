import React from 'react';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import MeterScreen from './src/screens/MeterScreen';

export default function App() {
  return <SafeAreaProvider><MeterScreen /></SafeAreaProvider>;
}
