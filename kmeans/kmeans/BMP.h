struct DIBHeader
{
	long DIBHeaderSize;
	long imageWidth;
	long imageHeight;
	short planes;
	short bitsPerPixel;
	long compression;
	long imageSize;
	long xPixelsPerMeter;
	long yPixelsPerMeter;
	long colorsInColorTable;
	long importantColorCount;

	DIBHeader(int w, int h)
	{
		imageWidth = w;
		imageHeight = h;
		DIBHeaderSize = 40;
		planes = 1;
		bitsPerPixel = 24;
		compression = 0;
		imageSize = 0;
		xPixelsPerMeter = 0;
		yPixelsPerMeter = 0;
		colorsInColorTable = 0;
		importantColorCount = 0;
	}
};