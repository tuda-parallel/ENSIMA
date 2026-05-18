#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/*
   Function to import data from a CSV file.
   The file is an OpenForm result file (labeled values)
   The imported data are written to a new file in CSV format

   Examples for the different considered labels :
   0: *Title,Front Door Results [180.0, 2.00, 2.0, 0.06, 200]
   1: "Thickness Change %-Min",00059871,   -20.3
   2: "Thickness Change %-Max",00060865,     2.5
   3: "Total Distance to Reference-Min",00096037,      0.0E+00
   4: "Total Distance to Reference-Max",00096781,      2.5E+00
   5: *Formability,Inadequate Stretch,Wrinkling, Wrinkling Tendency,
                         Safe,Risk of Cracks,Severe Thinning,Cracks
           0.0,    0.9,     6.6,    92.4,     0.0,     0.0,     0.0
*/

#define FALSE     0
#define TRUE      1
#define NEWLINE   '\n'
#define NO_ENTRY  '*'

#define SIZE     132    /* Characters in an input line */
#define MXDATA   16     /* Max. entries per input line */
#define MYHUGE   1.e+30 /* Place holder for NaN        */
#define EPS03    1.e-03 /* Comparison tolerance for    */
#define EPS06    1.e-06 /* Comparison tolerance for    */

#define DBGLEV   0

int ReadCSVData (char *FileNameIn, int limit, char *String,  int  nString,
		       char StrSTA[][30], char StrEND[][30], int *nStrEnt);
int AppendToFile(char *FileNameOu, char *String);
int read_line   (FILE *fp,   char *line, int limit, int *nChar, int *nSepar);
int find_line   (char *line, char *string);
int GatherData  (char *line, int nString,
		       char StrSTA[][30], char StrEND[][30], int *nStrEnt,
		                             int *StrNUM, double *pEntry);

int main()
{
  int    nStrEnt[6] = {0};    /* Number of entries in a label/string   */
  int    nString;             /* Number of currently considered labels */ 
  int    nStrlen, i, j;
  char   FileNameIn[512];
  char   FileNameOu[512];
  char   String[SIZE+1] =  " ";
  char   StrSTA[6][30]  = {" "};
  char   StrEND[6][30]  = {" "};

  /* Store labels to be looked for (currently 6 labels 30 char long) */

  strncpy(StrSTA[0],    "[\0", 2);
  strncpy(StrEND[0],    "]\0", 2);
  nStrEnt[0] = 5;

  nStrlen = (int)strlen("Thickness\0");
  strncpy(StrSTA[1],    "Thickness\0", nStrlen);
  nStrlen = (int)strlen("%-Min\0");
  strncpy(StrEND[1],    "%-Min\0", nStrlen);
  nStrEnt[1] = 1;

  nStrlen = (int)strlen("Thickness\0");
  strncpy(StrSTA[2],    "Thickness\0", nStrlen);
  nStrlen = (int)strlen("%-Max\0");
  strncpy(StrEND[2],    "%-Max\0", nStrlen);
  nStrEnt[2] = 1;

  nStrlen = (int)strlen("Total Distance\0");
  strncpy(StrSTA[3],    "Total Distance\0", nStrlen);
  nStrlen = (int)strlen("-Min\0");
  strncpy(StrEND[3],    "-Min\0", nStrlen);
  nStrEnt[3] = 1;

  nStrlen = (int)strlen("Total Distance\0");
  strncpy(StrSTA[4],    "Total Distance\0", nStrlen);
  nStrlen = (int)strlen("-Max\0");
  strncpy(StrEND[4],    "-Max\0", nStrlen);
  nStrEnt[4] = 1;

  nStrlen = (int)strlen("Formability\0");
  strncpy(StrSTA[5],    "Formability\0", nStrlen);
  nStrlen = (int)strlen("Cracks\0");
  strncpy(StrEND[5],    "Cracks\0", nStrlen);
  nStrEnt[5] = 7;

  nString = 6;
  if(DBGLEV > 0)
  { 
    printf("\n --> main: Implemented Labels\n");
    for(i = 0; i < nString; i++)
      printf("\n label %-2d: starts at %s and ends at %s (n = %-2d)", i,
                                       StrSTA[i], StrEND[i], nStrEnt[i]);
    printf("\n");
  }

  /* Import data from CSV file */

  printf("\nPlease enter file name of the CSV file: \n");
  scanf("%s", FileNameIn);

  if(!ReadCSVData(FileNameIn, SIZE, String, nString, StrSTA, StrEND, nStrEnt))
    exit(-1);
  printf("\n --> main : Imported values: %s\n", String);

  /* Append data to a CSV file */

  printf("\nPlease enter file name of the export file: \n");
  scanf("%s", FileNameOu);

  if(!AppendToFile(FileNameOu, String)) exit(-1);

  if(DBGLEV > 0) printf("\n --> main : Successfully finished\n");

  exit(0);
}

/***************************************************************************/

int ReadCSVData(char *FileNameIn, int limit, char *String,  int  nString,
		      char StrSTA[][30], char StrEND[][30], int *nStrEnt)
{
  FILE   *fp;
  double pEntry[7]  = {0.};   /* Numerical entries in a line (label)  */
  double Export[17] = {0.};   /* Line with all entries to be exported */
  int    nChar,  nSepar, nLines, nEntry, StrNUM, nStrlen;
  int    i, j, k, n;
  char   line[SIZE+1] = " ", cEntry[SIZE+1] = " ", c[1];

  if((fp = fopen(FileNameIn, "r")) == NULL) return(FALSE);

  /* Read line and find its correspondence in the list of labels */
  /* then store the gathered data in pEntry                      */

  k = 0;
  nLines = 0;
  do
  {
    read_line(fp, line, limit, &nChar, &nSepar);
    nLines++;   

    GatherData(line, nString, StrSTA, StrEND, nStrEnt, &StrNUM, pEntry);
    if(StrNUM != -1)
    {
      if(DBGLEV > 0) printf("\n Line %2d content: %s ", nLines, line);
      if(DBGLEV > 1) printf("\n Line correspond to Label type %2d ", StrNUM);
    }

    /* Gathering data from line */

    if(StrNUM == 0)         /* Label Type 0: find position of data in line */
    {
      i = strcspn(line, StrSTA[0]);    /* Position of '[' */
      j = strcspn(line, StrEND[0]);    /* Position of ']' */

      for(nEntry = 0; nEntry < nStrEnt[0]; nEntry++)  /* Loop over entries */
      {
        n=0; i++;
        do                                            /* Until next comma  */
	{
          cEntry[n] = line[i];
          n++; i++;
        } while((line[i] != ',') && (i != j));
        cEntry[n] = '\0';

        pEntry[nEntry] = atof(cEntry);               /* Convert to digits  */

        if(DBGLEV > 0) 
          printf("\n Entry %2d: string %s / digit %8.2e", nEntry,cEntry, 
	                                                  pEntry[nEntry]);
        Export[k++] = pEntry[nEntry];
      }
    } 
                               /* Label Types 1-4: find data at line's end */
    else if((StrNUM >= 1) && (StrNUM <= 4))
    {
      j = (int)strlen(line)-1;
      i = j;
      do                                              /* Until last comma  */
      {
        i--;
      } while((line[i] != ',') && (line[i] != ' '));

      n=0; i++;
      do
      {
        cEntry[n] = line[i];
        n++; i++;
      } while(i < j);
      cEntry[n] = '\0';

      nEntry    = 0;
      pEntry[nEntry] = atof(cEntry);                 /* Convert to digits  */

      if(DBGLEV > 0) 
        printf("\n Entry %2d: string %s / digit %8.2e", nEntry,cEntry, 
	                                                pEntry[nEntry]);
      Export[k++] = pEntry[nEntry];
    }

    else if(StrNUM == 5)                 /* Label Type 5: import next line */
    {
      read_line(fp, line, limit, &nChar, &nSepar);
      nLines++;   
      if(DBGLEV > 0) printf("\n Line %2d content: %s ", nLines, line);

      i=0;
      for(nEntry = 0; nEntry < nStrEnt[5]; nEntry++)  /* Loop over entries */
      {
        n=0; i++;
        do                                            /* Until next comma  */
	{
          if(line[i] != ' ') { cEntry[n] = line[i]; n++; }
          i++;
        } while((line[i] != ',') && (line[i] != '\0'));
        cEntry[n] = '\0';

        pEntry[nEntry] = atof(cEntry);               /* Convert to digits  */

        if(DBGLEV > 0) 
          printf("\n Entry %2d: string %s / digit %8.2e", nEntry,cEntry, 
  	                                                  pEntry[nEntry]);
        Export[k++] = pEntry[nEntry];
      }
      
    }

  } while(!find_line(line, String));
  if(DBGLEV > 0) printf("\n\n Number of significant lines: %3d\n", nLines);
  
  sprintf( String,     "%5.1f,",     Export[0]);
  sprintf((String+6),  "%4.2f,",     Export[1]);
  sprintf((String+11), "%4.1f,",     Export[2]);
  sprintf((String+16), "%4.2f,",     Export[3]);
  sprintf((String+21), "%3d,",  (int)Export[4]);
  sprintf((String+25), "%5.1f,",     Export[5]);
  sprintf((String+31), "%4.1f,",     Export[6]);

  i = 36;
  for(k = 9; k < 16; k++)
  { 
    sprintf((String+i),"%4.1f,", Export[k]);
    i += 5;
  }
  sprintf((String+i),  "%4.1f,", Export[8]);

  nStrlen = (int)strlen(String);
  strncpy((String+nStrlen+1), "\0", 1);

  if(DBGLEV > 0) printf("\n Current content of String : %s", String);

  fclose(fp);

  return(TRUE);
}

/***************************************************************************/
int AppendToFile(char *FileNameOu, char *String)
{
  FILE   *fp;
  int    numRow;

  if((fp = fopen(FileNameOu, "a")) == NULL) return(FALSE);

  /*  fseek(fp, 0L, 2); */
  numRow = 1111;
  fprintf(fp, "%4d,%s\n", numRow, String);

  fclose(fp);

  return(TRUE);
}
/***************************************************************************/
int read_line(FILE *fp, char *line, int limit, int *nChar, int *nSepar)
{
  char c[1];
  int  nCount1, nCount2;

  nCount1 = 0;    /* Character counter */
  nCount2 = 0;    /* Separator counter */
  while((c[0] = (char)fgetc(fp)) != NEWLINE) /* read until end of line */
  {
    if((c[0] == (char)EOF) && (DBGLEV > 1)) 
      printf("\n --> read_line: End-Of-File reached\n");

    /*    printf("\n Character read : %c", c[0]); */

    nCount1++;                               /* count characters */ 
    if(nCount1 < limit)                      /* string overflow? */
      line[nCount1-1] = c[0];
    else
    {
      printf("\n --> read_line: line too long\n");
      return(FALSE);
    }
    if(c[0] == ',') nCount2++;               /* count separators */ 
  }

  if(nCount1 <= 1)
  {
    if(DBGLEV > 1) printf("\n --> read_line: End-Of-Line reached");
    line[nCount1] = '\0';
    *nSepar  = -1;
    return(FALSE);
  }

  line[nCount1] = '\0';
  *nChar   = nCount1;
  *nSepar  = nCount2;

  if(DBGLEV >= 2)
  {
    printf("\n Imported line     : %s " , line);
    printf("\n Number characters : %-8d", nCount1);
    printf("\n Number separators : %-8d", nCount2);
  }

  return(TRUE);
}

/***************************************************************************/
int GatherData (char *line, int nString,
		      char StrSTA[][30], char StrEND[][30], int *nStrEnt,
		                             int *StrNUM, double *pEntry)
{
  int i, pSTA, pEND; 

/*if(DBGLEV != 0) printf("\n --> GatherData : %s ", line); */

  *StrNUM = -1;
  for(i = 0; i < nString; i++)
  {
    if((strstr(line,StrSTA[i]) != NULL) && (strstr(line,StrEND[i]) != NULL))
      *StrNUM = i;
  }

  return(TRUE);
}

/***************************************************************************/
int find_line(char *line, char *string)
{
  while((*line == *string) && (*string != '\0'))
    line++, string++;

  if(*string == '\0') return(TRUE);
  return(FALSE);
}

/***************************************************************************/

