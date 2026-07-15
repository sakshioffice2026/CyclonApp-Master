using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Database
{
    public class Revision
    {
        public int Id { get; set; }
        public int CycloneDesignId { get; set; }  // Must match FK column name
        public int RevisionNumber { get; set; }
        public string Description { get; set; }
        public DateTime CreatedAt { get; set; }
        public virtual CycloneDesign CycloneDesign { get; set; }
    }
}
