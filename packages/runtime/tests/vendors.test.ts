import { describe, expect, it } from 'vitest';
import { ericsson } from '../src/vendors';

describe('ericsson pattern', () => {
  it.each([
    'E300MM000032',
    'E300MM999001',
    'E300MM999023',
  ])('strictRegex matches %s', (s) => {
    expect(ericsson.strictRegex.test(s)).toBe(true);
  });

  it.each([
    'e303mm500942',     // lowercase
    'E103MM500942',     // 1 is not in [39]
    'E303MR500942',     // MR not MM
    'E300MM99002',      // 5 digits not 6
    'E300MM0000322',    // 7 digits
    'X303MM500942',     // X prefix
  ])('strictRegex rejects %s', (s) => {
    expect(ericsson.strictRegex.test(s)).toBe(false);
  });

  it('partialRegex extracts from longer string', () => {
    const m = 'noise E300MM000032 more'.match(ericsson.partialRegex);
    expect(m?.[0]).toBe('E300MM000032');
  });
});
