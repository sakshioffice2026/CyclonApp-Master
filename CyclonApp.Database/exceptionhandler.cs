using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Database
{
     public class exceptionhandler
    {
        [Key]
        public int id { get; set; }
        public string classname { get; set; }
        public string methodname { get; set; }
        public string error { get; set; }
        public DateTime datetime { get; set; }
        public int userid { get; set; }

    }
}
