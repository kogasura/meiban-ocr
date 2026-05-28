import { describe, expect, it } from 'vitest';
import { BLANK_IDX, CHARSET, NUM_CLASSES } from '../src/constants';

describe('constants', () => {
  it('CHARSET is 36 unique chars', () => {
    expect(CHARSET).toHaveLength(36);
    expect(new Set(CHARSET).size).toBe(36);
  });

  it('CHARSET matches digits then uppercase', () => {
    expect(CHARSET).toBe('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ');
  });

  it('BLANK_IDX equals len(CHARSET)', () => {
    expect(BLANK_IDX).toBe(36);
    expect(NUM_CLASSES).toBe(37);
  });
});
