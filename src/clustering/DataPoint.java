package clustering;
import static clustering.G.*;

public class DataPoint {
	//tailored to PRS and RKM needs
	private double[] coord=new double[D]; //D=dimensions of the point
	private int cID=-1 /*centroid id*/, saved_cID=-1, ID/*point uid*/;
	private int n; //number of added points
	private double dist, silhouette;
	
	public DataPoint() {}
	public DataPoint( double...coord ) { 
		if( coord.length!=D ) throw new IllegalArgumentException();
		System.arraycopy( coord,0,this.coord,0,D );
	}
	public DataPoint( DataPoint p ) {
		System.arraycopy( p.coord,0,this.coord,0,D );
		this.ID=p.ID;
	}
	
	public double[] getCoord() { 
		DataPoint dp=new DataPoint( this.coord );
		return dp.coord;
	}//getCoord
	public void setCoord( double...coord ) {
		System.arraycopy( coord,0,this.coord,0,D );
	}//setCoord
	public int getCID() { return cID; }
	public void setCID( int cID ) { this.cID=cID; }
	public void saveCID() { saved_cID=cID; }
	public void restoreCID() { cID=saved_cID; }
	public int getN() { return n; }
	public void setN( int n ) { this.n=n; }
	public double getDist() { return dist; }
	public void setDist( double dist ) { this.dist=dist; }
	public double getSilhouette() { return silhouette; }
	public void setSilhouette( double s ) { silhouette=s; }
	public int getID() { return ID; }
	public void setID( int ID ) { this.ID=ID; }
	
	public void reset() {
		for( int d=0; d<D; ++d ) coord[d]=0.0D;
		n=0;
	}//reset
	
	public double distance( DataPoint p ) {//Euclidean
		double s=0;
		for( int c=0; c<D; ++c )
			s=s+(this.coord[c]-p.coord[c])*(this.coord[c]-p.coord[c]);
		return Math.sqrt(s);
	}//distance
	
	public boolean nullPoint() { 
		for( int d=0; d<D; ++d )
			if ( coord[d]!=0 ) return false;
		return true;
	}//nullPoint
	
	public void add( DataPoint p ) {
		//to coordinates of this are added the coordinates of p
		for( int d=0; d<D; ++d )
			this.coord[d]=this.coord[d]+ p.coord[d];
		n++; //count this addition
	}//add
	
	public void sub( DataPoint p ) {
		//to coordinates of this are subtracted the coordinates of p
		for( int d=0; d<D; ++d )
			this.coord[d]=this.coord[d]-p.coord[d];
		n--; //count this subtraction		
	}//sub
	
	public void mean() {
		for( int d=0; d<D; ++d ) 
			coord[d]=coord[d]/n;
	}//mean
	
	public boolean equals( Object o ) {
		//deep equals
		if( !(o instanceof DataPoint) ) return false;
		if( o==this ) return true;
		DataPoint p=(DataPoint)o;
		for( int c=0; c<D; ++c )
			if( Math.abs(this.coord[c]-p.coord[c])>THR ) return false;
		return true;
	}//equals
	
	public int hashCode() {
		return java.util.Arrays.hashCode( coord );
	}//hashCode
	
	public String toString() {
		return ""+java.util.Arrays.toString(coord);
	}//toString
	
}//DataPoint
