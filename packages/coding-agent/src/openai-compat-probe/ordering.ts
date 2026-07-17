export function compareProbeStrings(left: string, right: string): number {
	return left < right ? -1 : left > right ? 1 : 0;
}

export function compareProbeObjectKeys(left: string, right: string): number {
	return compareProbeStrings(left.toLowerCase(), right.toLowerCase()) || compareProbeStrings(left, right);
}
