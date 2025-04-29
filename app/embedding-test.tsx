import React from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { Stack } from 'expo-router';
import { NativeTextEmbedder } from '@/components/NativeTextEmbedder';

export default function EmbeddingTestScreen() {
  return (
    <ScrollView style={styles.container}>
      <Stack.Screen 
        options={{ 
          title: 'Embedding Test',
          headerTitleAlign: 'center'
        }} 
      />
      <View style={styles.content}>
        <NativeTextEmbedder />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  content: {
    flex: 1,
    padding: 16,
  },
});