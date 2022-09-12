package clustering;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Stream;

public final class G {//Globals
	//tailored to PRS and RKM needs
	public static final int N=100000; //numerosity of the dataset
	public static final int D=2; //dimensions - aka nr of attributes per data point
	public static final int K=100; //number of clusters - var to be possibly set by program
    public static final int T=5000; //max number of iterations or steps
    public static final int RUNS=100; //number of repetition runs - useful e.g. for RKM
    public static final double THR=1E-10; //threshold
    public static int it=0; //iterations count
    
    public static String datasetName="BIRCH1"; //directory
    static String fileName="birch1.txt"; //dataset file
	
    public static boolean OUTPUT=true, INIT_CENTROIDS=true, INIT_GT=true, 
    					  INIT_PARTITIONS=false, PARALLEL=true, 
    		              DEBUG=true, PERSIST=true;
    static enum INIT{ RANDOM } //Omissis
    public static INIT INIT_METHOD=INIT.RANDOM;

    public static DataPoint[] dataset=new DataPoint[N];
    public static DataPoint[] centroids=new DataPoint[K];
    public static DataPoint[] newCentroids=new DataPoint[K];
    public static DataPoint[] gt=new DataPoint[K];
    
    static {
    	for( int i=0; i<K; ++i ) 
    		newCentroids[i]=new DataPoint();
    }
       
    public static void initialize_centroids() throws IOException{
   		switch( INIT_METHOD ) {
   			case RANDOM: random(); break;
   			//OMISSIS
    		//further alternatives to be added here
    	}
    }//initialize_centroids
    
    static void random() throws IOException {
		//initialize centroids as random data points chosen in the dataset
		boolean chosen[]=new boolean[N];
		for( int k=0; k<K; ++k ) {
			int h=0;
			do{
			  h=(int)(Math.random()*(N-1)); //random index in [0,N-1]
			  if( chosen[h] ) continue;
			  chosen[h]=true;
			}while( !chosen[h] );
			double[] coord=dataset[h].getCoord();
			centroids[k]=new DataPoint( coord ); 
		}
    }//random
       
    public static void show_centroids() {
    	for( int i=0; i<K; ++i )
    		System.out.println(centroids[i]);
    }//show_centroids
    
    public static void save_centroids( DataPoint[] centres ) {
    	for( int k=0; k<K; ++k ) {
    		centroids[k]=centres[k];
    		centroids[k].setN( centres[k].getN() );
    	}
    }//save_centroids
     
    public static double silhouette() throws IOException{
    	long startS=System.currentTimeMillis();
    	Stream<DataPoint> pStream=Stream.of( dataset ).parallel();
    	if( PARALLEL ) pStream=pStream.parallel();
    	pStream
    		.map( p -> {
    			int k=p.getCID(); //p's belonging cluster id
    			int nk=centroids[k].getN();
    			double icd=0; //internal cluster distances of p
    			double[] xcd=new double[K]; //external cluster distances of p
    			for( int i=0; i<N; ++i ) {
    				//if( dataset[i].getDist()!=-1 ) {
    				if( dataset[i].getCID()==k )
    					icd=icd+p.distance(dataset[i]);
    				else {
    					xcd[dataset[i].getCID()]=
    						xcd[dataset[i].getCID()]+p.distance(dataset[i]);
    				}
    			}
    			double axi=icd/(nk-1);
    			double bxi=Double.MAX_VALUE;
    			for( int c=0; c<K; ++c )
    				if( c!=k && xcd[c]/centroids[c].getN()<bxi ) 
    					bxi=xcd[c]/centroids[c].getN();
    			p.setSilhouette( (bxi-axi)/Math.max(bxi,axi) );
    			return p;
    		 } )
    		.forEach( p->{} );
    	  	
    	pStream=Stream.of( dataset );
    	if( PARALLEL ) pStream=pStream.parallel();
    	DataPoint p=pStream
    			//.filter( dp->dp.getDist()!=-1 )
    			.reduce( new DataPoint(), 
    					(p1,p2)->{ 
    						DataPoint r=new DataPoint();
    						r.setSilhouette( p1.getSilhouette()+p2.getSilhouette() ); 
    						return r; 
    					} );
    	long endS=System.currentTimeMillis();
    	System.out.println("Silhouette PET="+(endS-startS)+" msec");
    	if( OUTPUT )
    		println("Silhouette PET="+(endS-startS)+" msec");
    	return p.getSilhouette()/N;
    }//pSilhouette
           
    public static double elbow() {
    	//Find the total average sum of squared distances within clusters
    	double[] wss=new double[K]; //within cluster squared sum
    	for( int i=0; i<N; ++i ) {
    		int k=dataset[i].getCID();
    		double d=dataset[i].distance(centroids[k]);
    		wss[k]=wss[k]+d*d;
    	}
    	double tot_awss=0; //total average of wss
    	for( int k=0; k<K; ++k ) {
    		tot_awss=tot_awss+wss[k]/centroids[k].getN();
    	}
    	return tot_awss/K;
    }//elbow
       
    public static double SSE() {
    	Stream<DataPoint> pStream=Stream.of( dataset );
    	if( PARALLEL ) pStream=pStream.parallel();
    	DataPoint s=pStream
    		.map( p ->{
    			int k=p.getCID();
    			double d=p.distance(centroids[k]);
    			p.setDist( d*d );
    			return p;
    		} )
    		.reduce( new DataPoint(), 
    				(p1,p2)->{ DataPoint ps=new DataPoint(); 
    						   ps.setDist( p1.getDist()+p2.getDist() ); 
    						   return ps; } );
    	return s.getDist();
    }//SSE
    
    public static double nMSE() {
    	Stream<DataPoint> pStream=Stream.of( dataset );
    	if( PARALLEL ) pStream=pStream.parallel();
    	DataPoint s=pStream
    		.map( p ->{
    			int k=p.getCID();
    			double d=p.distance(centroids[k]);
    			p.setDist( d*d );
    			return p;
    		} )
    		.reduce( new DataPoint(), 
    				(p1,p2)->{ DataPoint ps=new DataPoint(); 
    				           ps.setDist( p1.getDist()+p2.getDist() ); 
    				           return ps; } );
    	return s.getDist()/(N*D);
    }//nMSE
    
    public static double SSE( final DataPoint[] centroids ) {
    	Stream<DataPoint> pStream=Stream.of( dataset );
    	if( PARALLEL ) pStream=pStream.parallel();
    	DataPoint s=pStream
    		.map( p ->{
    			int k=p.getCID();
    			double d=p.distance(centroids[k]);
    			p.setDist( d*d );
    			return p;
    		} )
    		.reduce( new DataPoint(), 
    				(p1,p2)->{ DataPoint ps=new DataPoint(); 
    						   ps.setDist( p1.getDist()+p2.getDist() ); 
    						   return ps; } );
    	return s.getDist();
    }//SSE
    
    public static double nMSE( final DataPoint[] centroids ) {
    	Stream<DataPoint> pStream=Stream.of( dataset );
    	if( PARALLEL ) pStream=pStream.parallel();
    	DataPoint s=pStream
    		.map( p ->{
    			int k=p.getCID();
    			double d=p.distance(centroids[k]);
    			p.setDist( d*d );
    			return p;
    		} )
    		.reduce( new DataPoint(), 
    				(p1,p2)->{ DataPoint ps=new DataPoint(); 
    				           ps.setDist( p1.getDist()+p2.getDist() ); 
    				           return ps; } );
    	return s.getDist()/(N*D);
    }//nMSE
      
    public static double ad2nc() {//average distance to nearest centroids
    	double[] s=new double[K];
    	for( int i=0; i<N; ++i ) {
    		int k=dataset[i].getCID();
    		s[k]=s[k]+dataset[i].distance(centroids[k]);
    	}
    	double S=0;
    	for( int k=0; k<K; ++k ) {
    		s[k]=s[k]/centroids[k].getN();
    		S=S+s[k];
    	}
    	S=S/K;
    	return S;
    }//ad2nc
    
    public static double wb_index() {
    	//origin of wb_index: w(ithin and b(etween squared indexes
    	//wb_index gets a minimal value as the number of clusters gets optimized
    	
    	//find the center of the data set
    	DataPoint xc=new DataPoint();
    	for( int i=0; i<N; ++i )
    		xc.add( dataset[i] );
    	xc.mean();
    	//then calculate SSW - sum of squares within clusters
    	double ssw=0;
    	for( int i=0; i<N; ++i ) {
    		double d=dataset[i].distance( centroids[dataset[i].getCID()] );
    		ssw=ssw+d*d;
    	}
    	//then calculate SSB - sum of squares between clusters
    	double ssb=0;
    	for( int k=0; k<K; ++k ) {
    		double d=centroids[k].distance(xc);
    		ssb=ssb+centroids[k].getN()*d*d;
    	}
    	return K*(ssw/ssb);
    }//wb_index
    
    public static double bcss() { //must be maximized
    	//Between Cluster Sum of Squared distances - Witten, Tibshirani 2010
    	//1st: detect global centroid
    	DataPoint gc=new DataPoint();
    	for( int i=0; i<N; ++i )
    		gc.add( dataset[i] );
    	gc.mean();
    	//2nd: detect sum of squared distances of points to gc
    	double ssdgc=0D;
    	for( int i=0; i<N; ++i ) {
    		double d=dataset[i].distance(gc);
    		ssdgc=ssdgc+d*d;	
    	}
    	return ssdgc-SSE();
    }//bcss
    
    public static int CI( DataPoint[] c1, DataPoint[] c2 ) {
    	//Franti's Centroid Index
    	
    	int m[]=new int[K]; //mapping
    	for( int k=0; k<K; ++k ) m[k]=-1;
    	
    	//mapping c1->c2
    	for( int i=0; i<K; ++i ) {
    		//map c1[i] onto nearest c2[j]
    		double md=Double.MAX_VALUE;
    		int mj=-1; //minimum prototype in c2 mapped from c1[i]
    		for( int j=0; j<K; ++j ) {
    			double d=c1[i].distance(c2[j]);
    			if( d*d<md ) { mj=j; md=d*d; }
    		}
    		m[i]=mj;	
    	}
    	
    	int CL12=0;
    	for( int j=0; j<K; ++j ) {
    		//is c2[j] an orphan?
    		boolean orphan=true;
    		for( int i=0; i<K; ++i )
    			if( m[i]==j ) orphan=false; //a mapping exists!
    		CL12 = orphan ? CL12+1 : CL12;
    	}  	

    	for( int k=0; k<K; ++k ) m[k]=-1; //m reset 
    	
    	//mapping c2->c1
    	for( int i=0; i<K; ++i ) {
    		//map c2[i] onto nearest c1[j]
    		double md=Double.MAX_VALUE;
    		int mj=-1; //minimum prototype in c1 mapped from c2[i]
    		for( int j=0; j<K; ++j ) {
    			double d=c2[i].distance(c1[j]);
    			if( d*d<md ) { mj=j; md=d*d; }
    		}
    		m[i]=mj;	
    	}
    	
    	int CL21=0;
    	for( int j=0; j<K; ++j ) {
    		//is c1[j] an orphan?
    		boolean orphan=true;
    		for( int i=0; i<K; ++i )
    			if( m[i]==j ) orphan=false;
    		CL21 = orphan ? CL21+1 : CL21;
    	}
    	
    	return Math.max( CL12,CL21 ); 
    	
    }//CI
    
    private static int shared( Set<Integer> s1, Set<Integer> s2 ) {
    	int s=0;
    	for( int ip: s1 ) 
    		if( s2.contains( ip ) ) s++;
    	return s;
    }//shared
    
    public static int GCI( Set<Integer>[] p1, Set<Integer>[] p2 ) {
    	//Generalized Centroid Index - Franti
    	//matrix cm used to store shared points
    	int[][] cm=new int[K][K]; //all zeros initially
    	
    	for( int i=0; i<K; ++i ) //costs: O(K^2)
    		for( int j=0; j<K; ++j ) {
    			cm[i][j]=shared( p1[i], p2[j] );
    		}
    	
    	//P1 and P2 are partitions - aka arrays of K sets/lists of points
    	int m[]=new int[K]; //mapping
    	for( int k=0; k<K; ++k ) m[k]=-1;
    	
    	//mapping p1->p2
    	for( int i=0; i<K; ++i ) {
    		//map p1[i] onto "nearest" p2[j]
    		//that is, the cluster with most shared elements with p1[i]
    		int max=Integer.MIN_VALUE, mj=-1;
    		for( int j=0; j<K; ++j ) {
    			if( cm[i][j]>max ) { max=cm[i][j]; mj=j; }		
    		}
    		m[i]=mj;
    	}
    	
    	int CL12=0;
    	for( int j=0; j<K; ++j ) {
    		//is p2[j] an orphan?
    		boolean orphan=true;
    		for( int i=0; i<K; ++i )
    			if( m[i]==j ) orphan=false; //a mapping exists!
    		CL12 = orphan ? CL12+1 : CL12;
    	}  	

    	for( int k=0; k<K; ++k ) m[k]=-1; //m reset 
    	
    	//mapping p2->p1
    	for( int i=0; i<K; ++i ) {
    		//map p2[i] onto "nearest" p1[j]
    		int max=Integer.MIN_VALUE, mj=-1;
    		for( int j=0; j<K; ++j ) {
    			if( cm[j][i]>max ) { max=cm[j][i]; mj=j; }		
    		}
    		m[i]=mj;	
    	}
    	
    	int CL21=0;
    	for( int j=0; j<K; ++j ) {
    		//is c1[j] an orphan?
    		boolean orphan=true;
    		for( int i=0; i<K; ++i )
    			if( m[i]==j ) orphan=false;
    		CL21 = orphan ? CL21+1 : CL21;
    	}
    	
    	return Math.max( CL12,CL21 ); 
    	
    }//GCI

    private static double jaccard_distance( Set<Integer> s1, Set<Integer> s2 ) {
    	Set<Integer> tmp=new HashSet<>(s1);
    	tmp.retainAll(s2);
    	int intersection_size=tmp.size();
    	tmp.clear(); tmp.addAll(s1);
    	tmp.addAll(s2);
    	int union_size=tmp.size();
    	double j=1.0-(double)intersection_size/union_size; //Jaccard distance
    	return j;
    }//jaccard_distance
    
    public static int jGCI( Set<Integer>[] p1, Set<Integer>[] p2 ) {
    	//Generalized Centroid Index - Franti
    	//matrix cm is used to store Jaccard distances
    	double[][] cm=new double[K][K]; //all zeros initially
    	
    	for( int i=0; i<K; ++i ) //costs: O(K^2)
    		for( int j=0; j<K; ++j )
    			cm[i][j]=jaccard_distance( p1[i], p2[j] );
    	
    	//P1 and P2 are partitions - aka arrays of K sets/lists of points
    	int m[]=new int[K]; //mapping
    	for( int k=0; k<K; ++k ) m[k]=-1;
    	
    	//mapping p1->p2
    	for( int i=0; i<K; ++i ) {
    		//map p1[i] onto "nearest" p2[j] according to Jaccard distance
    		double min=Double.MAX_VALUE; int mj=-1;
    		for( int j=0; j<K; ++j ) {
    			if( cm[i][j]<min ) { min=cm[i][j]; mj=j; }		
    		}
    		m[i]=mj;
    	}
    	
    	int CL12=0;
    	for( int j=0; j<K; ++j ) {
    		//is p2[j] an orphan?
    		boolean orphan=true;
    		for( int i=0; i<K; ++i )
    			if( m[i]==j ) orphan=false; //a mapping exists!
    		CL12 = orphan ? CL12+1 : CL12;
    	}  	

    	for( int k=0; k<K; ++k ) m[k]=-1; //m reset 
    	
    	//mapping p2->p1
    	for( int i=0; i<K; ++i ) {
    		//map p2[i] onto "nearest" p1[j]
    		double min=Double.MAX_VALUE; int mj=-1;
    		for( int j=0; j<K; ++j ) {
    			if( cm[j][i]<min ) { min=cm[j][i]; mj=j; }		
    		}
    		m[i]=mj;	
    	}
    	
    	int CL21=0;
    	for( int j=0; j<K; ++j ) {
    		//is c1[j] an orphan?
    		boolean orphan=true;
    		for( int i=0; i<K; ++i )
    			if( m[i]==j ) orphan=false;
    		CL21 = orphan ? CL21+1 : CL21;
    	}
    	
    	return Math.max( CL12,CL21 ); 
    	
    }//jGCI
    
    public static void load_dataset() throws IOException{
		BufferedReader br=new BufferedReader( 
			new FileReader("d:\\datasets\\"+datasetName+"\\"+fileName) );
		int i=0;
		try {
			for(;;) {
				if( i==N ) break;
				String linea=br.readLine();
				if( linea==null ) break;
				linea=linea.trim();
				String[] coord=linea.split("\\s+");
				if( coord.length!=D ) {
					System.out.println("coord.length="+coord.length+" inattesa sulla linea "+i+" del dataset");
					throw new IllegalArgumentException("Bad dataset file.");
				}
				double[] coo=new double[D];
				for( int j=0; j<D; ++j ) coo[j]=Double.valueOf(coord[j]);
				dataset[i]=new DataPoint( coo );
				dataset[i].setID(i);
				i++;
			}
		}finally { 
			br.close(); 
		}   
    }//load_dataset
    
    public static void load_centroids() throws IOException{
    	if( INIT_CENTROIDS ) 
    		initialize_centroids();
    	else {
    		//centroids already existing
    		BufferedReader br=new BufferedReader( 
    				new FileReader("d:\\datasets\\"+datasetName+"\\centroids.txt") );
    		int i=0;
    		try {
    			for(;;) {
    				if( i==K ) break;
    				String linea=br.readLine();
    				if( linea==null ) break;
    				linea=linea.trim();
    				String[] coord=linea.split("\\s+");
    				if( coord.length!=D ) {
    					System.out.println("coord.length="+coord.length+" unexpected on line "+i+" of the centroids file");
    					throw new IllegalArgumentException("Bad centroids file.");
    				}
    				double[] c=new double[coord.length];
    				for( int j=0; j<D; ++j ) c[j]=Double.valueOf( coord[j] );
    				centroids[i]=new DataPoint( c );
    				i++;
    			}
    		}finally { 
    			br.close(); 
    		}  
    	}
    }//load_centroids
        
    public static void load_gt() throws IOException{
    	if( INIT_GT ) {
    		BufferedReader br=new BufferedReader( 
    				new FileReader("d:\\datasets\\"+datasetName+"\\gt.txt") );
    		int i=0;
    		try {
    			for(;;) {
    				if( i==K ) break;
    				String linea=br.readLine();
    				if( linea==null ) break;
    				linea=linea.trim();
    				String[] coord=linea.split("\\s+");
    				if( coord.length!=D ) {
    					System.out.println("coord.length="+coord.length+" unexpected on line "+i+" of the gt file");
    					throw new IllegalArgumentException("Bad gt file.");
    				}
    				double[] c=new double[coord.length];
    				for( int j=0; j<D; ++j ) c[j]=Double.valueOf( coord[j] );
    				gt[i]=new DataPoint( c );
    				i++;
    			}
    		}finally { 
    			br.close(); 
    		}  
    	}
    }//load_gt

    public static Set<Integer>[] iPartition, fPartition;
    static {
    	if( INIT_PARTITIONS ) {
    		iPartition=new HashSet[K];
    		fPartition=new HashSet[K];
    		for( int k=0; k<K; ++k ) {
    			iPartition[k]=new java.util.HashSet<>();
    			fPartition[k]=new java.util.HashSet<>();
    		}
    	}
    }
    
    public static void load_partitions() throws IOException{
    	if( INIT_PARTITIONS ) {
    		BufferedReader br=new BufferedReader( 
    				new FileReader("d:\\datasets\\"+datasetName+"\\labels.txt") );
    		int i=0; //dataset point index
    		try {
    			for(;;) {
    				if( i==N ) break;
    				String linea=br.readLine();
    				if( linea==null ) break;
    				int label=Integer.parseInt(linea);
    				if( label<1 || label>K ) throw new IOException("Bad labels file.");
    				iPartition[ label-1 ].add( i );
    				i++;
    			}
    		}finally { 
    			br.close(); 
    		}  
    	}
    }//load_partitions
    
	public static void final_partitions() {
		for( int k=0; k<K; ++k ) fPartition[k].clear();
		for( int i=0; i<N; ++i ) {
			int k=dataset[i].getCID();
			fPartition[k].add(i);
		}
	}//final_partitions
    
    //simple output handling
    static PrintWriter pw=null;
    public static void open( String nomeFile ) throws IOException{
    	//output file for storing execution results
    	pw=new PrintWriter( new FileWriter(nomeFile) );
    }//open
    
    public static void close() {
    	pw.close();
    }//close
    
    public static void println( String line ){
    	pw.println(line);
    }//println
    
    public static void print( String s ){
    	pw.print(s);
    }//println
    
    public static void println(){
    	pw.println();
    }//println
    
    public static void persist_centroids() {
    	println("Centroids:");
    	for( int k=0; k<K; ++k ) {
    		double[] coord=centroids[k].getCoord();
			for( int d=0; d<D; ++d ) {
				print( ""+coord[d] );
				if( d<D-1 ) print(" ");
			}
			println();    		
    	}
    	println();
    }//persist_centroids
    
    public static void persist_partitions() {
    	println("Partition sizes: ");
    	int[] psize=new int[K];
    	for( int i=0; i<N; ++i ) {
    		int k=dataset[i].getCID();
    		psize[k]++;
    	}
    	int tot=0;
    	for( int k=0; k<K; ++k ) {
    		println("psize["+k+"]="+psize[k]);
    		tot=tot+psize[k];
    	}
    	println("Total: "+tot);
    	println();
    	println("Clustering labels from 0 to "+(K-1)+" :");
    	for( int i=0; i<N; ++i )
    		println(""+dataset[i].getCID());
    	println();   	
    }//persist_partitions
    
    public static void persist_solution(){
    	persist_centroids();
    	persist_partitions();
    }//persist

}//G
